import json
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, DatasetDict, load_dataset
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer
import evaluate_models


def main():
    filenames = ["llm-goal-scope/data/qa_dataset/original/no_rag/allocationQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/original/no_rag/comparativeAssertionsQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/original/no_rag/functionalUnitQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/original/no_rag/intendedApplicationQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/original/no_rag/productQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/original/no_rag/studyReasonsQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/original/no_rag/systemBoundaryQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/original/no_rag/targetAudienceQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/allocationQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/comparativeAssertionsQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/functionalUnitQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/intendedApplicationQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/productQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/studyReasonsQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/systemBoundaryQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/targetAudienceQA.jsonl",
                 ]

    # for each dataset
    for k in filenames:
        data = []
        with open(k, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        if len(data) > 0:
            # convert to dataset
            dataset = load_dataset('json', data_files=k) # shuffle dataset before splitting
            dataset = dataset.shuffle(seed=42)
            covariance(dataset, k)


def covariance(dataset, dataset_name):
    # calculate covariance
    # process data
    classes = [class_ for class_ in dataset['train'][0]['all_labels'].split("; ") if class_]
    if len(classes) > 1:
        class2id = {class_: id for id, class_ in enumerate(classes)}

        # model setup
        model_path = 'microsoft/deberta-v3-small'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenized_dataset = dataset.map(lambda example: evaluate_models.preprocess_function(example, classes, class2id, tokenizer))

        # covariance calculations
        df = pd.DataFrame(tokenized_dataset["train"])
        new_columns_df = pd.DataFrame(df['labels'].tolist(), index=df.index, columns=classes)
        labels_array = new_columns_df.values
        covariance_matrix = np.cov(labels_array.T)
        correlation = new_columns_df.corr()

        # plot covariance
        correlation_plotting(classes, correlation, dataset_name)
        covariance_plotting(classes, covariance_matrix, dataset_name)


def covariance_plotting(classes, covariance_matrix, dataset_name):
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    sns.heatmap(covariance_matrix,
                annot=True,  # Show the covariance values on the heatmap
                fmt='.2f',  # Format the annotation values to two decimal places
                cmap='viridis',  # Choose a colormap (e.g., 'viridis', 'coolwarm', 'RdBu')
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Covariance Matrix Heatmap')
    # open output
    fpath = "llm-goal-scope/data/qa_dataset/results/"
    dataset_name = dataset_name.split(".")[0]
    dataset_name = dataset_name.split("/")[2:]
    dataset_name = "_".join(dataset_name)
    os.makedirs("llm-goal-scope/data/qa_dataset/results/" + dataset_name + "/", exist_ok=True)
    plt.savefig(fpath + dataset_name + "/covariance.png", dpi=300)
    plt.show()


def correlation_plotting(classes, correlation_matrix, dataset_name):
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    sns.heatmap(correlation_matrix,
                annot=True,  # Show the covariance values on the heatmap
                fmt='.1f',  # Format the annotation values to two decimal places
                cmap='viridis',  # Choose a colormap (e.g., 'viridis', 'coolwarm', 'RdBu')
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Correlation Matrix Heatmap')
    # open output
    fpath = "llm-goal-scope/data/qa_dataset/results/"
    dataset_name = dataset_name.split(".")[0]
    dataset_name = dataset_name.split("/")[2:]
    dataset_name = "_".join(dataset_name)
    os.makedirs("llm-goal-scope/data/qa_dataset/results/" + dataset_name + "/", exist_ok=True)
    plt.savefig(fpath + dataset_name + "/correlation.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
