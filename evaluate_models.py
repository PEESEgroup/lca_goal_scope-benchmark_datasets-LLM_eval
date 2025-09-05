import constants
from langchain_community.vectorstores import FAISS
import json
import evaluate
import numpy as np
from evaluate import evaluator
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer


def preprocess_function(example, classes, class2id, tokenizer):
    text = f"{example['title']}.\n{example['content']}"
    all_labels = example['all_labels'].split(', ')
    labels = [0. for i in range(len(classes))]
    for label in all_labels:
        label_id = class2id[label]
        labels[label_id] = 1.

    example = tokenizer(text, truncation=True)
    example['labels'] = labels
    return example


def sigmoid(x):
   return 1/(1 + np.exp(-x))


def compute_metrics(eval_pred):

   predictions, labels = eval_pred
   predictions = sigmoid(predictions)
   predictions = (predictions > 0.5).astype(int).reshape(-1)
   clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
   return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))


def eval_models(dataset, dataset_name):
    # from: https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification
    dataset = load_dataset('knowledgator/events_classification_biotech')

    classes = [class_ for class_ in dataset['train'].features['label 1'].names if class_]
    class2id = {class_: id for id, class_ in enumerate(classes)}
    id2class = {id: class_ for class_, id in class2id.items()}

    model_path = 'microsoft/deberta-v3-small'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized_dataset = dataset.map(preprocess_function)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=len(classes),
        id2label=id2class, label2id=class2id,
        problem_type="multi_label_classification"
    )
    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics,
    )

    trainer.train()

    # for each model, we need to run an evaluation
    model_names = ["distilbert-base-uncased-distilled-squad"]
    for i in model_names:
        # as label classification
        print("evaluating dataset", dataset_name, " using model", i)
        task_evaluator = evaluator("text-classification")
        eval_results = task_evaluator.compute(
            model_or_pipeline=i,
            input_column="question",
            label_column="answers",
            label_mapping={},
            data=dataset,
            metric=evaluate.combine(["accuracy", "recall", "precision", "f1"])
        )


        """        # as qa
        # TODO: filter by type???
        task_evaluator = evaluator("question-answering")
        print("evaluating dataset", dataset_name, " using model", i)
        eval_results = task_evaluator.compute(
            model_or_pipeline=i,
            data=dataset,
            metric="squad", #TODO: is the metric having difficulty due to the use of "squad". what others are available?
            strategy="bootstrap",
            n_resamples=30
        )"""


        """
        # TODO: try text-generation
        task_evaluator = evaluator("text-generation")
        print("evaluating dataset", dataset_name, " using model", i)
        eval_results = task_evaluator.compute(
            model_or_pipeline=i,
            data=dataset,
            metric="squad",
            strategy="bootstrap",
            input_column="question",
            label_column="answers",
            n_resamples=30
        )
        """

        """
        # for text2text-generation, which may or may not be what we want
        task_evaluator = evaluator("text2text-generation")
        print("evaluating dataset", dataset_name, " using model", i)
        eval_results = task_evaluator.compute(
            model_or_pipeline=i,
            data=dataset,
            metric="squad",
            strategy="bootstrap",
            n_resamples=30
        )

        print(dataset_name, eval_results)
        """

        print(dataset_name, eval_results)


if __name__ == "__main__":
    # load vdb information
    embeddings = constants.EMBED_MODEL
    vdb = FAISS.load_local(
        constants.VDB_LOCATION, embeddings, allow_dangerous_deserialization=True)
    print("vdb loaded")

    # load qa dataset
    filenames = ["data/recalculated/qa_dataset.jsonl", "data/qa_dataset.jsonl", "data/recalculated/rag_qa_dataset.jsonl", "data/rag_qa_dataset.jsonl"]
    for k in filenames:
        data = []
        with open(k, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        # convert to dataset
        # TODO: combine question and context into a single entry
        #TODO: rerun rag code
        dataset = Dataset.from_list(data)
        data = load_dataset("squad", split="validation[:2]")
        print(data[0])
        print(dataset[0])
        print("dataset loaded")
        eval_models(dataset, k)

