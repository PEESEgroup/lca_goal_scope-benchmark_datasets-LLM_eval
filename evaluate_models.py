import evaluate
import numpy as np
import csv
import pandas as pd
import shutil
import matplotlib
import collections
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import gc
from typing import Optional
import os
from datasets import load_dataset, DatasetDict
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, \
    AutoTokenizer
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay


class CustomLossTrainer(Trainer):
    """
    Custom loss trainer class wrapping the asymmetric loss function
    """

    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Use the loss_fn passed in, or default to a NEW instance
        self.loss_fn = loss_fn if loss_fn is not None else AsymmetricLossOptimized()

    def compute_loss(self, model, inputs, num_items_in_batch: Optional[torch.Tensor] = None, return_outputs=False):
        """
        computes the loss from logits and labels
        :param model: model of interest
        :param inputs: tensor inputs
        :param num_items_in_batch: number of items in batch
        :param return_outputs: outputs of the method, defaults to false
        :return: loss, outputs is return_outputs is true
        """
        # Assume your inputs include "labels" and your model returns logits.
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # get the batch size
        batch_size = labels.size(0)

        # Compute the custom loss using your loss function.
        loss = self.loss_fn(logits, labels) / batch_size

        return (loss, outputs) if return_outputs else loss


class AsymmetricLossOptimized(nn.Module):
    """
    Asymmetric Loss Function class
    """
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    # https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf
    # https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        """
        Initialization function
        :param gamma_neg: negative gamma value
        :param gamma_pos: positive gamma value
        :param clip: clip value
        :param eps: epsilon value
        :param disable_torch_grad_focal_loss: focal loss
        """
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


def preprocess_function(examples, classes, class2id, tokenizer):
    """
    preprocess data
    :param example: a row of data
    :param classes: list of classes
    :param class2id: dict of classes and labels
    :param tokenizer: tokenizer model
    :return: return example with updated labels
    """
    texts = [f"{t}.\n{c}" for t, c in zip(examples['title'], examples['context'])]
    batch_size = len(examples['labels'])
    num_classes = len(classes)
    labels_matrix = np.zeros((batch_size, num_classes), dtype=np.float32)

    for idx, label_list in enumerate(examples['labels']):
        for label in label_list:
            if label in class2id:
                labels_matrix[idx, class2id[label]] = 1.0
    
    tokenized = tokenizer(texts, truncation=True)
    tokenized['labels'] = labels_matrix.tolist()
    return tokenized


def sigmoid(x):
    """
    Simple sigmoid function
    :param x: predictions
    :return: sigmoid of predictions
    """
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    """
    compute metrics function
    :param eval_pred: tuple of predictions and labels
    :return: take the sigmoid of the predictions and then calculate model performance metrics
    """
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    clf_metrics = evaluate.combine(["accuracy"])
    return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))


def train_model(model, tokenized_dataset, tokenizer, data_collator, dataset_name, model_path, loss_fn=None):
    """
    train the LLM models
    :param model: LLM model
    :param tokenized_dataset: tokenized dataset
    :param tokenizer: tokenizer LLM
    :param data_collator: data collator
    :param dataset_name: name of the dataset
    :param model_path: path to storage location of the LLM for checkpoints and trained models
    :return: custom loss trainer
    """
    # create necessary filepaths
    if loss_fn is not None:
        suffix = "-bce"
    else:
        suffix = ''

    checkpoint_path = "llm-goal-scope/data/checkpoints/" + model_path + "/" + dataset_name + suffix
    final_model_path = "llm-goal-scope/data/trained_model/" + model_path + "/" + dataset_name + suffix

    # training parameters
    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=15,  # try 15
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
        loss_fn=loss_fn
    )
    # training
    trainer.train()

    # save model to an S3 bucket to save space???
    # trainer.save_model(final_model_path)
    log_history_df = pd.DataFrame(trainer.state.log_history)
    plotting(log_history_df, dataset_name, model_path)
    return trainer


def eval_metrics(tokenized_dataset, trainer, classes, dataset_name, fpath):
    """
    evaluate trained model on dataset
    :param tokenized_dataset: tokenized dataset
    :param trainer: custom trainer
    :param classes: list of all classes
    :param dataset_name: name of dataset
    :param fpath: path of output storage
    :return: datasets of all predictions and errors
    """
    # predict step
    predictions_output = trainer.predict(tokenized_dataset["test"])
    validation_output = trainer.predict(tokenized_dataset["valid"])

    # confusion matrix converts probabilities based on a threshold value and then take the sigmoid of the outputs
    eval_metrics = predictions_output.metrics
    val_metrics = validation_output.metrics
    multilabel_indicators = (1 / (1 + np.exp(-predictions_output.predictions)))
    threshold = 0.7
    multilabel_preds = multilabel_indicators > threshold
    
    plt.figure()
    plt.hist(multilabel_indicators.flatten(), bins=20)
    plt.title(f'Raw Logit Predictions for {dataset_name}')
    plt.savefig(f'{fpath}/Raw Logit Predictions for {dataset_name}.png', dpi=300)
    plt.close('all')

    cm = multilabel_confusion_matrix(predictions_output.label_ids, multilabel_preds)
    ap_scores = []
    for i, cm in enumerate(cm):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix for {classes[i]} class for ' + str(dataset_name) + ' with threshold ' + str(threshold))
        plt.savefig(fpath + f'/Confusion Matrix for {classes[i].replace("/", "")} class.png', dpi=300)
        plt.close('all')

    # calculate hamming accuracy
    h_loss = hamming_loss(predictions_output.label_ids, multilabel_preds)
    hamming_score = 1 - h_loss
    eval_metrics["accuracy (hamming loss)"] = f"{hamming_score:.4f}"

    # record data
    if predictions_output.metrics:
        with open(fpath + '/test_metrics.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["Test Metrics:"])
            w.writerows(eval_metrics.items())
            w.writerow(["Validation Metrics:"])
            w.writerows(val_metrics.items())
            print(f"Saved Metrics for {dataset_name}")


    # get the validation dataset to match val_logits length
    valid_dataset = tokenized_dataset["valid"]
    test_dataset = tokenized_dataset["test"]

    # Create the Test Predictions DataFrame
    prediction_df = pd.DataFrame({
        'context': test_dataset['context'],
        'test_logits': predictions_output.predictions.tolist(),
        'true_labels': predictions_output.label_ids.astype(int).tolist(),
        'classes': [classes] * len(test_dataset)
    })

    # Create the Validation Predictions DataFrame (Optional, but keeps your data safe)
    validation_df = pd.DataFrame({
        'context': valid_dataset['context'],
        'val_logits': validation_output.predictions.tolist(),
        'true_labels': validation_output.label_ids.astype(int).tolist(),
        'classes': [classes] * len(valid_dataset)
    })
    
    # Save validation predictions to disk right here so you don't lose them
    validation_df.to_csv(fpath + "/validation_predictions.csv", index=False)
    prediction_df.to_csv(fpath + "/test_predictions.csv", index=False)


def eval_models(dataset, dataset_name, ablation_loss_fn=None, suffix=""):
    """
    evaluate model loop
    :param dataset: dataset of interest
    :param dataset_name: name of dataset
    :return: N/A
    """
    # from: https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification
    # bad practice, but because all the labels are in each row of the dataset, things can be trained
    if "all_labels" in dataset["train"][0]:
        classes = [class_ for class_ in dataset['train'][0]['all_labels'].split("; ") if class_]
        class2id = {class_: id for id, class_ in enumerate(classes)}
        id2class = {id: class_ for class_, id in class2id.items()}

        model_paths = ['microsoft/deberta-v3-small', 'microsoft/deberta-v3-base', 
                        'microsoft/deberta-v3-large',
                       # these models are confirmed to work
                       "google-bert/bert-base-uncased", "FacebookAI/roberta-large",
                       "climatebert/distilroberta-base-climate-f", "ESGBERT/EnvironmentalBERT-base"]

        # train and eval loop
        for model_path in model_paths:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenized_dataset = dataset.map(
                lambda examples: preprocess_function(examples, classes, class2id, tokenizer),
                batched=True,
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
            dataset_path = dataset_path + suffix
            fpath = "/home/sagemaker-user/llm-goal-scope/data/dataset/results/" + dataset_path + "/" + model_path
            os.makedirs(fpath, exist_ok=True)

            # train model
            if ablation_loss_fn is not None:
                trainer = train_model(model, tokenized_dataset, tokenizer, data_collator, dataset_path, model_path, ablation_loss_fn)
            else:
                trainer = train_model(model, tokenized_dataset, tokenizer, data_collator, dataset_path, model_path)

            # eval model
            print("test dataset evaluation")
            eval_metrics(tokenized_dataset, trainer, classes, dataset_path, fpath)

            # cleaning up after model
            print(f"Cleaning up after {model_path}...")

            # delete the big GPU objects and force garbage collection
            del model
            del trainer
            gc.collect()

            # clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # remove these folders to save space
            for dir_path in ["llm-goal-scope/data/checkpoints/", "llm-goal-scope/data/trained_model/"]:
                try:
                    shutil.rmtree(dir_path)
                    print(f"{dir_path} and contents successfully deleted.")
                except FileNotFoundError:
                    print(f"The system cannot find {dir_path}")
                except PermissionError:
                    print("Permission denied. Ensure files are not open in another program.")

    # else there is no data in the datset
    else:
        print("dataset missing:", str(dataset_name))


def plotting(log_history_df, dataset_name, model_name):
    """
    Plot model loss function
    :param log_history_df: dataframe containing the history of training
    :param dataset_name: name of the dataset
    :param model_name: name of the model
    :return: N/A loss plot
    """
    train_logs = log_history_df[log_history_df['loss'].notna()]
    eval_logs = log_history_df[log_history_df['eval_loss'].notna()]
    # list filepath and create directory to store the image
    fpath = f"/home/sagemaker-user/llm-goal-scope/data/dataset/results/{dataset_name}/{model_name}"
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

    # ignoring comparative assertion, intended application, study reasons, and target audience as Hestia does not have that data
    # ignoring recalculated allocation because it only uses the economic label
    filenames = [
        "llm-goal-scope/data/dataset/original/no_rag/Functional Unit.jsonl",
        "llm-goal-scope/data/dataset/original/no_rag/System Boundary.jsonl",
        "llm-goal-scope/data/dataset/original/no_rag/Allocation.jsonl",
        "llm-goal-scope/data/dataset/original/no_rag/Product.jsonl",
        "llm-goal-scope/data/dataset/standardized/no_rag/Functional Unit.jsonl",
        "llm-goal-scope/data/dataset/standardized/no_rag/Product.jsonl",
        "llm-goal-scope/data/dataset/standardized/no_rag/System Boundary.jsonl",
        "llm-goal-scope/data/dataset/original/rag/Allocation.jsonl",
        "llm-goal-scope/data/dataset/original/rag/Functional Unit.jsonl",
        "llm-goal-scope/data/dataset/original/rag/Product.jsonl",
        "llm-goal-scope/data/dataset/original/rag/System Boundary.jsonl",
        "llm-goal-scope/data/dataset/standardized/rag/Functional Unit.jsonl",
        "llm-goal-scope/data/dataset/standardized/rag/Product.jsonl",
        "llm-goal-scope/data/dataset/standardized/rag/System Boundary.jsonl",
        "llm-goal-scope/data/dataset/original/rag/Functional Unitn-retrieved-10.jsonl",
        "llm-goal-scope/data/dataset/original/rag/Functional Unitn-retrieved-20.jsonl",
        "llm-goal-scope/data/dataset/original/rag/Functional Unitn-top-1.jsonl",
        "llm-goal-scope/data/dataset/original/rag/Functional Unitn-top-5.jsonl",
        "llm-goal-scope/data/dataset/original/rag/Functional Unittemp-033.jsonl",
        "llm-goal-scope/data/dataset/original/rag/Functional Unittemp-090.jsonl",
        "llm-goal-scope/data/dataset/original/rag/Functional Unittokens-128.jsonl",
        "llm-goal-scope/data/dataset/original/rag/Functional Unittokens-512.jsonl"
    ]

    # for each dataset
    for k in filenames:
        if k == "llm-goal-scope/data/dataset/original/no_rag/Functional Unit.jsonl":
            ablation = True
        else:
            ablation = False
        
        # Do all ablation studies
        if ablation:
            # load the dataset
            dataset = load_dataset('json', data_files=k)  # shuffle dataset before splitting
            dataset = dataset.shuffle(seed=42)

            for s in ['cycle', 'site', 'source', 'loss_fn']:
                if s in ['cycle', 'site', 'source']:
                    # calculate dynamic train/test/validation splits
                    # find the frequency of the rarest item
                    column_data = dataset['train'][s]
                    unique_counts = collections.Counter(column_data)
                    min_frequency = min(unique_counts.values())
                    
                    # Guardrail: If an item only appears 1 or 2 times, it cannot be split 3 ways.
                    if min_frequency < 3:
                        print(f"Column '{s}' has a rare item appearing only {min_frequency} time(s).")
                        print("Stratification into 3 splits is mathematically impossible. Skipping stratification...")
                        continue  # if it is impossible, then nothing can be done. This is ablation test anyways...
                    else:
                        # max items we can take from the rarest class for validation (and test)
                        max_items_per_slice = math.floor(min_frequency / 3)
                        
                        # calculate the maximum percentage this represents for the rarest class
                        max_fraction_per_slice = max_items_per_slice / min_frequency
                        
                        # Cap it at a reasonable global limit (e.g., max 20% test, 20% valid) so training set doesn't get small on well-populated datasets.
                        if max_fraction_per_slice > 0.20:
                            print("Stratification into 3 splits would require training data size <60%. Skipping stratification...")
                            continue
                        test_fraction = min(max_fraction_per_slice, 0.20)
                        train_fraction = 1.0 - test_fraction*2
                        print(f"Train fraction for s is {train_fraction}, test/validation fraction is {test_fraction}")

                    train_testvalid = dataset['train'].train_test_split(test_size=test_fraction*2, seed=42, stratify_by_column=s)
                    # Split the 10% test + valid in half test, half valid
                    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42, stratify_by_column=s)
                    # gather everyone if you want to have a single DatasetDict
                    train_test_valid_dataset = DatasetDict({
                        'train': train_testvalid['train'],
                        'test': test_valid['test'],
                        'valid': test_valid['train']})

                    print(str(k), "dataset loaded stratified on " + s)
                    eval_models(train_test_valid_dataset, k, suffix=s)
                if s == 'loss_fn':
                    bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
                    # 80% train, 20% test + validation
                    train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=42)
                    # Split the 10% test + valid in half test, half valid
                    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
                    # gather everyone if you want to have a single DatasetDict
                    train_test_valid_dataset = DatasetDict({
                        'train': train_testvalid['train'],
                        'test': test_valid['test'],
                        'valid': test_valid['train']})

                    print(str(k), "dataset loaded with BCE loss")
                    eval_models(train_test_valid_dataset, k, ablation_loss_fn=bce_loss, suffix=s)

        # Do things once without ablation
        # load the dataset
        dataset = load_dataset('json', data_files=k)  # shuffle dataset before splitting
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

        
