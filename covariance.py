import json
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer
import evaluate_models


def main():
    filenames = ["data/qa_dataset/original/no_rag/allocationQA.jsonl",
                 "data/qa_dataset/original/no_rag/comparativeAssertionsQA.jsonl",
                 "data/qa_dataset/original/no_rag/functionalUnitQA.jsonl",
                 "data/qa_dataset/original/no_rag/intendedApplicationQA.jsonl",
                 "data/qa_dataset/original/no_rag/productQA.jsonl",
                 "data/qa_dataset/original/no_rag/studyReasonsQA.jsonl",
                 "data/qa_dataset/original/no_rag/systemBoundaryQA.jsonl",
                 "data/qa_dataset/original/no_rag/targetAudienceQA.jsonl",
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
            covariance(dataset, k)


def covariance(dataset, dataset_name):
    # calculate covariance
    # process data
    dataset = DatasetDict({'train': dataset})
    classes = [class_ for class_ in dataset['train'][0]['all_labels'].split("; ") if class_]
    class2id = {class_: id for id, class_ in enumerate(classes)}
    id2class = {id: class_ for class_, id in class2id.items()}

    model_path = 'microsoft/deberta-v3-small'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized_dataset = dataset.map(lambda example: evaluate_models.preprocess_function(example, classes, class2id, tokenizer))

    df = pd.DataFrame(tokenized_dataset["train"])
    df_labels = df["labels"]
    labels_array = df_labels.values
    covariance_matrix = np.cov(labels_array.T)

    # plot covariance
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    sns.heatmap(covariance_matrix,
                annot=True,  # Show the covariance values on the heatmap
                fmt='.2f',  # Format the annotation values to two decimal places
                cmap='viridis',  # Choose a colormap (e.g., 'viridis', 'coolwarm', 'RdBu')
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Covariance Matrix Heatmap')

    # open output
    fpath = "data/qa_dataset/results/"
    dataset_name = dataset_name.split(".")[0]
    dataset_name = dataset_name.split("/")[2:]
    dataset_name = "_".join(dataset_name)
    os.makedirs("data/qa_dataset/results/" + dataset_name + "/", exist_ok=True)
    plt.savefig(fpath + dataset_name + "/loss.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
