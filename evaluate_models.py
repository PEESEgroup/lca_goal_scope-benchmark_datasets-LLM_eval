import constants
from langchain_community.vectorstores import FAISS
import json
import evaluate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, \
    AutoTokenizer


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

    reeee = load_dataset('knowledgator/events_classification_biotech', trust_remote_code=True)
    row = dataset["train"][0]

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
        training_args = TrainingArguments(
            output_dir="data/checkpoints/"+dataset_name,
            learning_rate=2e-5,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=3,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
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

        trainer.train()

        log_history_df = pd.DataFrame(trainer.state.log_history)
        train_logs = log_history_df[log_history_df['loss'].notna()]
        eval_logs = log_history_df[log_history_df['eval_loss'].notna()]
        fpath = "data/qa_dataset/results/"
        plotting(train_logs, eval_logs, dataset_name, fpath)
    else:
        print("dataset missing:", str(dataset_name))

    # TODO: evaluation


def plotting(train_logs, eval_logs, dataset_name, fpath):
    # Plotting Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_logs['step'], train_logs['loss'], label='Training Loss')
    plt.plot(eval_logs['step'], eval_logs['eval_loss'], label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time for ' + str(dataset_name))
    plt.legend()
    plt.grid(True)
    plt.savefig(fpath, dataset_name + "_loss.png", dpi=300)
    plt.show()

    # Plotting Accuracy (if 'accuracy' and 'eval_accuracy' are present in your logs)
    if 'accuracy' in train_logs.columns and 'eval_accuracy' in eval_logs.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(train_logs['step'], train_logs['accuracy'], label='Training Accuracy')
        plt.plot(eval_logs['step'], eval_logs['eval_accuracy'], label='Validation Accuracy')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy Over Time for '+ str(dataset_name))
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
    filenames = ["data/qa_dataset/recalculated/no_rag/productQA.jsonl", "data/qa_dataset.jsonl",
                 "data/recalculated/rag_qa_dataset.jsonl", "data/rag_qa_dataset.jsonl"]

    # for each dataset
    for k in filenames:
        data = []
        with open(k, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        # convert to dataset
        dataset = Dataset.from_list(data)

        # shuffle dataset before splitting
        dataset = dataset.shuffle(seed=42)

        # 80% train, 20% test + validation
        train_testvalid = dataset.train_test_split(test_size=0.2)
        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
        # gather everyone if you want to have a single DatasetDict
        train_test_valid_dataset = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'valid': test_valid['train']})

        print("dataset loaded")
        eval_models(train_test_valid_dataset, k)
