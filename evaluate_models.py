import constants
from langchain_community.vectorstores import FAISS
import json
import evaluate
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, \
    AutoTokenizer
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay


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


def eval_models(dataset, dataset_name):
    # from: https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification

    # reeee = load_dataset('knowledgator/events_classification_biotech', trust_remote_code=True)

    # bad practice, but because all the labels are in each row of the dataset, things can be trained
    if "all_labels" in dataset["train"][0]:
        classes = [class_ for class_ in dataset['train'][0]['all_labels'].split("; ") if class_]
        class2id = {class_: id for id, class_ in enumerate(classes)}
        id2class = {id: class_ for class_, id in class2id.items()}

        model_path = 'microsoft/deberta-v3-small'

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenized_dataset = dataset.map(lambda example: preprocess_function(example, classes, class2id, tokenizer))
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(classes),
            id2label=id2class,
            label2id=class2id,
            problem_type="multi_label_classification"
        )

        # update the dataset name
        dataset_name = dataset_name.split(".")[0]
        dataset_name = dataset_name.split("/")[2:]
        dataset_name = "_".join(dataset_name)
        os.makedirs("data/qa_dataset/results/" + dataset_name + "/", exist_ok=True)

        # training parameters
        training_args = TrainingArguments(
            output_dir="data/checkpoints/" + dataset_name,
            learning_rate=2e-5,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=3,
            num_train_epochs=10,
            weight_decay=0.01,
            eval_strategy="epoch",
            logging_strategy='epoch',
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        trainer = Trainer(
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
        log_history_df = pd.DataFrame(trainer.state.log_history)
        plotting(log_history_df, dataset_name)

        # eval
        print("test dataset evaluation")
        predictions_output = trainer.predict(tokenized_dataset["test"])

        # confusion matrix
        # convert probabilities based on a threshold value
        multilabel_indicators = (predictions_output.predictions > 1).astype(int)
        cm = multilabel_confusion_matrix(predictions_output.label_ids, multilabel_indicators)
        for i, cm in enumerate(cm):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
            disp.plot(cmap='Blues', values_format='d')
            plt.title(f'Confusion Matrix for {classes[i]} class for ' + str(dataset_name))
            plt.savefig("data/qa_dataset/results/" + dataset_name + f'/Confusion Matrix for {classes[i].replace("/", "")} class.png', dpi=300)
            plt.show()

        # record data
        if predictions_output.metrics:
            print(dataset_name)
            print("Metrics:", predictions_output.metrics)
            with open("data/qa_dataset/results/" + dataset_name + '/test_metrics.csv', 'w') as f:
                w = csv.writer(f)
                w.writerows(predictions_output.metrics.items())

    else:
        print("dataset missing:", str(dataset_name))


def plotting(log_history_df, dataset_name):
    train_logs = log_history_df[log_history_df['loss'].notna()]
    eval_logs = log_history_df[log_history_df['eval_loss'].notna()]
    fpath = "data/qa_dataset/results/"

    # Plotting Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_logs['step'], train_logs['loss'], label='Training Loss')
    plt.plot(eval_logs['step'], eval_logs['eval_loss'], label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time for ' + str(dataset_name))
    plt.legend()
    plt.grid(True)
    plt.savefig(fpath + dataset_name + "/loss.png", dpi=300)
    plt.show()

    # Plotting Accuracy (if 'accuracy' and 'eval_accuracy' are present in your logs)
    if 'accuracy' in train_logs.columns and 'eval_accuracy' in eval_logs.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(train_logs['step'], train_logs['accuracy'], label='Training Accuracy')
        plt.plot(eval_logs['step'], eval_logs['eval_accuracy'], label='Validation Accuracy')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy Over Time for ' + str(dataset_name))
        plt.legend()
        plt.grid(True)
        plt.savefig(fpath, dataset_name + "_accuracy.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    # load vdb information
    embeddings = constants.EMBED_MODEL
    vdb = FAISS.load_local(
        constants.VDB_LOCATION, embeddings, allow_dangerous_deserialization=True)
    print("vdb loaded")

    # load all datasets
    filenames = [#"data/qa_dataset/original/no_rag/allocationQA.jsonl",
    #              "data/qa_dataset/original/no_rag/comparativeAssertionsQA.jsonl",
    #              "data/qa_dataset/original/no_rag/functionalUnitQA.jsonl",
    #              "data/qa_dataset/original/no_rag/intendedApplicationQA.jsonl",
    #              "data/qa_dataset/original/no_rag/productQA.jsonl",
    #              "data/qa_dataset/original/no_rag/studyReasonsQA.jsonl",
    #              "data/qa_dataset/original/no_rag/systemBoundaryQA.jsonl",
    #              "data/qa_dataset/original/no_rag/targetAudienceQA.jsonl",

                 "data/qa_dataset/recalculated/no_rag/allocationQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/comparativeAssertionsQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/functionalUnitQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/intendedApplicationQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/productQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/studyReasonsQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/systemBoundaryQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/targetAudienceQA.jsonl",
                "data/qa_dataset/original/rag/rag_allocationQA.jsonl",
                "data/qa_dataset/original/rag/rag_comparativeAssertionsQA.jsonl",
                "data/qa_dataset/original/rag/rag_functionalUnitQA.jsonl",
                "data/qa_dataset/original/rag/rag_intendedApplicationQA.jsonl",
                "data/qa_dataset/original/rag/rag_productQA.jsonl",
                "data/qa_dataset/original/rag/rag_studyReasonsQA.jsonl",
                "data/qa_dataset/original/rag/rag_systemBoundaryQA.jsonl",
                "data/qa_dataset/original/rag/rag_targetAudienceQA.jsonl",
                 "data/qa_dataset/recalculated/rag/rag_allocationQA.jsonl",
                 "data/qa_dataset/recalculated/rag/rag_comparativeAssertionsQA.jsonl",
                 "data/qa_dataset/recalculated/rag/rag_functionalUnitQA.jsonl",
                 "data/qa_dataset/recalculated/rag/rag_intendedApplicationQA.jsonl",
                 "data/qa_dataset/recalculated/rag/rag_productQA.jsonl",
                 "data/qa_dataset/recalculated/rag/rag_studyReasonsQA.jsonl",
                 "data/qa_dataset/recalculated/rag/rag_systemBoundaryQA.jsonl",
                 "data/qa_dataset/recalculated/rag/rag_targetAudienceQA.jsonl"
                 ]

    # for each dataset
    for k in filenames:
        data = []
        with open(k, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        if len(data) > 0:
            # convert to dataset
            dataset = Dataset.from_list(data)

            # shuffle dataset before splitting
            dataset = dataset.shuffle(seed=42)

            # 80% train, 20% test + validation
            train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
            # Split the 10% test + valid in half test, half valid
            test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
            # gather everyone if you want to have a single DatasetDict
            train_test_valid_dataset = DatasetDict({
                'train': train_testvalid['train'],
                'test': test_valid['test'],
                'valid': test_valid['train']})

            print(str(k), "dataset loaded")
            eval_models(train_test_valid_dataset, k)
        else:
            # no data in the dataset
            print(str(k), "failed to load because there is no data")
