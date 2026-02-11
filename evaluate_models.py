import constants
import json
import evaluate
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import gc
import matplotlib as mpl
from typing import Optional
import os
from datasets import Dataset, load_dataset, DatasetDict
from sklearn.metrics import average_precision_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, \
    AutoTokenizer
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay


class CustomLossTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Use the loss_fn passed in, or default to a NEW instance
        self.loss_fn = loss_fn if loss_fn is not None else AsymmetricLossOptimized()

    def compute_loss(self, model, inputs, num_items_in_batch: Optional[torch.Tensor] = None, return_outputs=False):
        # Assume your inputs include "labels" and your model returns logits.
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # get the batch size
        # TODO: verify if this *needs* to be done because forward doesn't depend on batch size
        batch_size = labels.size(0)

        # Compute the custom loss using your loss function.
        loss = self.loss_fn(logits, labels) / batch_size

        return (loss, outputs) if return_outputs else loss


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    # https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf
    # https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, logits, labels):
        """"
        Parameters
        ----------
        logits: input logits
        labels: targets (multi-label binarized vector)
        """
        # using local variables instead of self.variable to avoid CUDA errors
        targets = labels
        anti_targets = 1 - labels

        # Calculating Probabilities
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss = loss + (anti_targets * torch.log(xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # Logic remains the same, just local variables
                    w = torch.pow(1 - (xs_pos * targets) - (xs_neg * anti_targets),
                                self.gamma_pos * targets + self.gamma_neg * anti_targets)
            else:
                w = torch.pow(1 - (xs_pos * targets) - (xs_neg * anti_targets),
                            self.gamma_pos * targets + self.gamma_neg * anti_targets)
            loss = loss * w

        return -loss.sum()


def preprocess_function(example, classes, class2id, tokenizer):
    text = f"{example['title']}.\n{example['context']}"
    all_labels = example['labels']
    labels = [0. for i in range(len(classes))]
    for label in all_labels:
        label_id = class2id[label]
        labels[label_id] = 1.

    example = tokenizer(text, truncation=True)
    example['labels'] = labels
    return example


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))


def train_model(model, tokenized_dataset, tokenizer, data_collator, dataset_name, model_path):
    # create necessary filepaths
    checkpoint_path = "llm-goal-scope/data/checkpoints/" + model_path + "/" + dataset_name
    final_model_path = "llm-goal-scope/data/trained_model/" + model_path + "/" + dataset_name

    # training parameters
    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        learning_rate=2e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=15, # try 15
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_strategy='epoch',
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
    )
    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # training
    trainer.train()

    #TODO: save model to an S3 bucket to save space???
    #TODO: clear out model checkpoints
    trainer.save_model(final_model_path)
    log_history_df = pd.DataFrame(trainer.state.log_history)
    plotting(log_history_df, dataset_name, model_path)
    return trainer


def eval_metrics(predictions_output, classes, dataset_name, fpath):
    # confusion matrix converts probabilities based on a threshold value and then take the sigmoid of the outputs
    eval_metrics = predictions_output.metrics
    multilabel_indicators = ((1 / (1 + np.exp(-predictions_output.predictions))) > 0.5).astype(int)
    plt.clf()
    plt.hist((1 / (1 + np.exp(-predictions_output.predictions))))
    plt.savefig(fpath + f'/Raw Logit Predictions for {dataset_name}.png', dpi=300)
    plt.show()

    cm = multilabel_confusion_matrix(predictions_output.label_ids, multilabel_indicators)
    ap_scores = []
    for i, cm in enumerate(cm):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix for {classes[i]} class for ' + str(dataset_name))
        plt.savefig(fpath + f'/Confusion Matrix for {classes[i].replace("/", "")} class.png', dpi=300)
        plt.show()

        # if there are no positive class in y_true, then precision is undefined and not included in the mean calculation
        if max(predictions_output.label_ids[:, i]) > 0:
            ap = average_precision_score(predictions_output.label_ids[:, i], multilabel_indicators[:, i])
        else:
            ap = np.nan
        ap_scores.append(ap)
        eval_metrics[f"Average Precision for Label {classes[i]}"] =  f"{ap:.4f}"
    print("saved confusion matrices")

    # Calculate Mean Average Precision (mAP)
    mAP = np.nanmean(ap_scores)
    eval_metrics["Mean Average Precision (mAP)"] = f"{mAP:.4f}"
    eval_metrics["dataset_name"] = f"{dataset_name}"
    eval_metrics["fpath"] = f"{fpath}"

    # record data
    if predictions_output.metrics:
        with open(fpath + '/test_metrics.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerows(eval_metrics.items())
            print("Saved Metrics for {dataset_name}:", eval_metrics)


def eval_models(dataset, dataset_name):
    # from: https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification
    # bad practice, but because all the labels are in each row of the dataset, things can be trained
    if "all_labels" in dataset["train"][0]:
        classes = [class_ for class_ in dataset['train'][0]['all_labels'].split("; ") if class_]
        class2id = {class_: id for id, class_ in enumerate(classes)}
        id2class = {id: class_ for class_, id in class2id.items()}

        model_paths = ['microsoft/deberta-v3-small','microsoft/deberta-v3-base', 'microsoft/deberta-v3-large',  # these models are confirmed to work
        "google-bert/bert-base-uncased", "FacebookAI/roberta-large", 
        "climatebert/distilroberta-base-climate-f", "ESGBERT/EnvironmentalBERT-base"]

        # train and eval loop
        for model_path in model_paths:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenized_dataset = dataset.map(
                lambda example: preprocess_function(example, classes, class2id, tokenizer),
                load_from_cache_file=False
            )
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(classes),
                id2label=id2class,
                label2id=class2id,
                problem_type="multi_label_classification"
            )

            # update the dataset name
            dataset_path = dataset_name.split(".")[0]
            dataset_path = dataset_path.split("/")[2:]
            dataset_path = "_".join(dataset_path)
            fpath = "/home/sagemaker-user/llm-goal-scope/data/qa_dataset/results/" + dataset_path + "/" + model_path
            os.makedirs(fpath, exist_ok=True)

            # train model
            trainer = train_model(model, tokenized_dataset, tokenizer, data_collator, dataset_path, model_path)

            # eval model
            print("test dataset evaluation")
            predictions_output = trainer.predict(tokenized_dataset["test"])
            eval_metrics(predictions_output, classes, dataset_path, fpath)

            # cleaning up after model
            print(f"Cleaning up after {model_path}...")
            
            # delete the big GPU objects and force garbage collection
            del model
            del trainer
            gc.collect()
            
            # clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    # else there is no data in the datset
    else:
        print("dataset missing:", str(dataset_path))


def plotting(log_history_df, dataset_name, model_name):
    train_logs = log_history_df[log_history_df['loss'].notna()]
    eval_logs = log_history_df[log_history_df['eval_loss'].notna()]
    # list filepath and create directory to store the image
    fpath = f"/home/sagemaker-user/llm-goal-scope/data/qa_dataset/results/{dataset_name}/{model_name}"
    os.makedirs(fpath, exist_ok=True)

    # Plotting Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_logs['step'], train_logs['loss'], label='Training Loss')
    plt.plot(eval_logs['step'], eval_logs['eval_loss'], label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time for ' + str(dataset_name))
    plt.legend()
    plt.grid(True)
    plt.savefig(fpath + "/loss.png", dpi=300)
    plt.show()
    print("loss plot saved")


if __name__ == "__main__":
    # load all datasets
    # apparently the debug and run configuration require different filepaths, so may need to remove the first directory
    filenames = ["llm-goal-scope/data/qa_dataset/original/no_rag/allocationQA.jsonl",  # have results
                 "llm-goal-scope/data/qa_dataset/original/no_rag/comparativeAssertionsQA.jsonl", # no dataset
                 "llm-goal-scope/data/qa_dataset/original/no_rag/functionalUnitQA.jsonl", # have results
                 "llm-goal-scope/data/qa_dataset/original/no_rag/intendedApplicationQA.jsonl", # no dataset
                 "llm-goal-scope/data/qa_dataset/original/no_rag/productQA.jsonl", # have results
                 "llm-goal-scope/data/qa_dataset/original/no_rag/studyReasonsQA.jsonl", # no dataset
                 "llm-goal-scope/data/qa_dataset/original/no_rag/systemBoundaryQA.jsonl", # ongoing
                 "llm-goal-scope/data/qa_dataset/original/no_rag/targetAudienceQA.jsonl", # no dataset
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/allocationQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/comparativeAssertionsQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/functionalUnitQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/intendedApplicationQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/productQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/studyReasonsQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/systemBoundaryQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/targetAudienceQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/original/rag/rag_allocationQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/original/rag/rag_comparativeAssertionsQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/original/rag/rag_functionalUnitQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/original/rag/rag_intendedApplicationQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/original/rag/rag_productQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/original/rag/rag_studyReasonsQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/original/rag/rag_systemBoundaryQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/original/rag/rag_targetAudienceQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/recalculated/rag/rag_allocationQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/recalculated/rag/rag_comparativeAssertionsQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/recalculated/rag/rag_functionalUnitQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/recalculated/rag/rag_intendedApplicationQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/recalculated/rag/rag_productQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/recalculated/rag/rag_studyReasonsQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/recalculated/rag/rag_systemBoundaryQA.jsonl",
                #  "llm-goal-scope/data/qa_dataset/recalculated/rag/rag_targetAudienceQA.jsonl"
                 ]

    # for each dataset
    for k in filenames:
        # load the dataset
        try:
            dataset = load_dataset('json', data_files=k) # shuffle dataset before splitting
            dataset = dataset.shuffle(seed=42)
            
            # 80% train, 20% test + validation
            train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=42)
            # Split the 10% test + valid in half test, half valid
            test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
            # gather everyone if you want to have a single DatasetDict
            train_test_valid_dataset = DatasetDict({
                'train': train_testvalid['train'],
                'test': test_valid['test'],
                'valid': test_valid['train']})

            print(str(k), "dataset loaded")
            eval_models(train_test_valid_dataset, k)

        except ValueError as e:
            print(str(k), "failed to load because there is no data")
