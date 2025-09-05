import constants
from langchain_community.vectorstores import FAISS
import json
import evaluate
import numpy as np
from evaluate import evaluator
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, \
    AutoTokenizer


def preprocess_function(example, classes, class2id, tokenizer):
    # TODO: replace with question and content from dataset files
    text = f"{example['title']}.\n{example['content']}"
    all_labels = example['all_labels']
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

    data = dataset
    dataset = load_dataset('knowledgator/events_classification_biotech', trust_remote_code=True)

    # a list of all of the unique classes
    classes = [class_ for class_ in dataset['train'].features['label 1'].names if class_]
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
    training_args = TrainingArguments(
        output_dir="my_awesome_model"+dataset_name,
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
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # TODO: evaluation


if __name__ == "__main__":
    # load vdb information
    embeddings = constants.EMBED_MODEL
    vdb = FAISS.load_local(
        constants.VDB_LOCATION, embeddings, allow_dangerous_deserialization=True)
    print("vdb loaded")

    # load all datasets
    filenames = ["data/recalculated/qa_dataset.jsonl", "data/qa_dataset.jsonl",
                 "data/recalculated/rag_qa_dataset.jsonl", "data/rag_qa_dataset.jsonl"]

    # for each dataset
    for k in filenames:
        data = []
        with open(k, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        # convert to dataset
        # TODO: combine question and context into a single entry
        # TODO: rerun rag code
        # TODO: test-train split
        dataset = Dataset.from_list(data)
        print("dataset loaded")
        eval_models(dataset, k)
