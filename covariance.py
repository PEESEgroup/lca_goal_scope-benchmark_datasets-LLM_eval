import json
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer
import evaluate_models
import collections
from collections import Counter


def main():
    # plot figures for dataset balance
    grouping("./data/dataset/original/no_rag/System Boundary.jsonl")

    # plot figures for the correlation and covariance of the labels in the datasets
    filenames = ["./data/dataset/original/no_rag/System Boundary.jsonl",
                 "./data/dataset/original/no_rag/Allocation.jsonl",
                 "./data/dataset/original/no_rag/Functional Unit.jsonl",
                 "./data/dataset/original/no_rag/Product.jsonl",
                 "./data/dataset/standardized/no_rag/Functional Unit.jsonl",
                 "./data/dataset/standardized/no_rag/Product.jsonl",
                 "./data/dataset/standardized/no_rag/System Boundary.jsonl",
                 ]

    # for each dataset
    for k in filenames:
        data = []
        with open(k, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        if len(data) > 0:
            # convert to dataset
            dataset = load_dataset('json', data_files=k)  # shuffle dataset before splitting
            dataset = dataset.shuffle(seed=42)
            covariance(dataset, k)


def grouping(k):
    """
    Given an input filename, produce a plot which gives measures of the balance and coverage of the dataset
    :param k: filename
    :return: N/A
    """
    # Load the dataset
    dataset = load_dataset('json',
                           data_files=k)  # each dataset shares same metadata, so that is invariant by dataset. Labels obviously differ

    # 1) Re-combine train, test, and validation splits into a single dataset
    if hasattr(dataset, 'keys') and len(dataset.keys()) > 1:
        full_dataset = concatenate_datasets([dataset[split] for split in dataset.keys()])
    else:
        full_dataset = dataset["train"] if hasattr(dataset, 'keys') else dataset

    full_dataset = full_dataset.shuffle(seed=42)

    # Include 'labels' in your columns audit list
    columns = ['cycle', 'site', 'source', 'grouped_cycle', "grouped_site", "grouped_source", 'random_cycle',
               "random_site", "random_source"]

    # Create a figure with subplots for each item (adjusted width for 4 subplots)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, s in enumerate(columns):
        ax = axes[idx]

        if 'grouped' in s:
            col = s.split("_")[1]
            # split the dataset with custom splitter enabling singletons
            split_dataset = custom_grouped_split(
                dataset=full_dataset,
                group_col=col,  # Choose which metadata column to balance around
                train_size=0.8,
                test_size=0.1,
                val_size=0.1,
                seed=42
            )

            # Multi-label / List handling:
            # Extract the universe of valid labels from the 'all_labels' column
            raw_all_labels = split_dataset['train']['all_labels'][0]
            valid_labels = raw_all_labels.split(";")
            valid_labels = [i.strip() for i in valid_labels]

            # Initialize counts for all valid labels to 0 (crucial for zero-coverage detection)
            unique_counts = {label: 0 for label in valid_labels}

            # Flatten and count occurrences across all rows
            for label_list in split_dataset['train']['labels']:
                for label in label_list:
                    if label in unique_counts:
                        unique_counts[label] += 1

            counts = list(unique_counts.values())
        elif "random" in s:
            train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=42)

            # Multi-label / List handling:
            # Extract the universe of valid labels from the 'all_labels' column
            raw_all_labels = train_testvalid['train']['all_labels'][0]
            valid_labels = raw_all_labels.split(";")
            valid_labels = [i.strip() for i in valid_labels]

            # Initialize counts for all valid labels to 0 (crucial for zero-coverage detection)
            unique_counts = {label: 0 for label in valid_labels}

            # Flatten and count occurrences across all rows
            for label_list in train_testvalid['train']['labels']:
                for label in label_list:
                    if label in unique_counts:
                        unique_counts[label] += 1

            counts = list(unique_counts.values())
        else:
            # Standard single-value categorical handling
            column_data = full_dataset[s]
            unique_counts = collections.Counter(column_data)
            counts = list(unique_counts.values())

        # 2) Calculate coverage and balance constraints
        min_freq = min(counts) if counts else 0
        mean_freq = np.mean(counts) if counts else 0
        std_freq = np.std(counts) if counts else 0

        # Coefficient of Variation (CV) measures balance
        cv = std_freq / mean_freq if mean_freq > 0 else float('inf')

        # Define thresholds
        COVERAGE_THRESHOLD = 3
        BALANCE_THRESHOLD = 1.5

        coverage_passed = min_freq >= COVERAGE_THRESHOLD
        balance_passed = cv <= BALANCE_THRESHOLD

        # Log status
        print(f"--- Column: {s} ---")
        print(
            f"  Min Frequency: {min_freq} (Required >= {COVERAGE_THRESHOLD}) -> {'PASS' if coverage_passed else 'FAIL'}")
        print(f"  Balance (CV):  {cv:.2f} (Required <= {BALANCE_THRESHOLD}) -> {'PASS' if balance_passed else 'FAIL'}")

        # Plotting the histogram
        ax.hist(counts, color='skyblue', edgecolor='black')

        status_label = f"Coverage: {'OK' if coverage_passed else 'FAIL'} | Balance: {'OK' if balance_passed else 'FAIL'}"
        ax.set_title(f"{s}\n({status_label})", fontsize=10)
        ax.set_xlabel(
            "Number of Samples per Group" if 'group' in s or "random" in s else "Number of Samples per Metadata Category")
        ax.set_ylabel("Frequency")
        ax.tick_params(axis='x', rotation=45)

    # share x axis for histogram plots
    master_ax = axes[5]
    for i in range(3, 9):
        axes[i].sharex(master_ax)

    plt.tight_layout()
    plt.savefig("./data/dataset/results/grouped_splits.png", dpi=300)
    plt.close()
    print("Plot successfully saved to './data/dataset/results/grouped_splits.png'")


def custom_grouped_split(dataset, group_col='source', train_size=0.8, test_size=0.1, val_size=0.1, seed=42):
    """
    Group-aware splitter that ensures all records sharing the same group_col
    (e.g., study ID, source, or paper) stay together in the same split
    to prevent sibling record leakage.
    """
    np.random.seed(seed)

    # Normalize probabilities to ensure they sum to 1.0
    total_ratio = train_size + val_size + test_size
    p_train = train_size / total_ratio
    p_val = val_size / total_ratio
    p_test = test_size / total_ratio
    probs = [p_train, p_val, p_test]

    # Extract group data
    group_data = dataset[group_col]

    # Map each unique group (e.g., study/source) to its corresponding row indices
    group_to_indices = {}
    for idx, val in enumerate(group_data):
        if val not in group_to_indices:
            group_to_indices[val] = []
        group_to_indices[val].append(idx)

    # Get the list of unique groups and shuffle them randomly
    unique_groups = list(group_to_indices.keys())
    np.random.shuffle(unique_groups)

    train_indices = []
    val_indices = []
    test_indices = []

    # Assign entire groups stochastically to train, validation, or test based on target ratios
    for group in unique_groups:
        indices = group_to_indices[group]

        # Randomly choose which split this entire group goes to based on probabilities
        chosen_split = np.random.choice(['train', 'val', 'test'], p=probs)

        if chosen_split == 'train':
            train_indices.extend(indices)
        elif chosen_split == 'val':
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

    # Shuffle internal indices within each split
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    split_dataset = DatasetDict({
        'train': dataset.select(train_indices),
        'validation': dataset.select(val_indices),
        'test': dataset.select(test_indices)
    })

    total_len = len(dataset)
    print(f"Grouped shuffle completed on group column '{group_col}':")
    print(f"  - Unique groups: {len(unique_groups)}")
    print(f"  - Train samples: {len(train_indices)} ({len(train_indices) / total_len * 100:.1f}%)")
    print(f"  - Validation samples: {len(val_indices)} ({len(val_indices) / total_len * 100:.1f}%)")
    print(f"  - Test samples: {len(test_indices)} ({len(test_indices) / total_len * 100:.1f}%)")

    return split_dataset


def covariance(dataset, dataset_name):
    # calculate covariance
    # process data
    classes = [class_ for class_ in dataset['train'][0]['all_labels'].split("; ") if class_]
    if len(classes) > 1:
        class2id = {class_: id for id, class_ in enumerate(classes)}

        # model setup
        model_path = 'microsoft/deberta-v3-small'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenized_dataset = dataset.map(
            lambda example: evaluate_models.preprocess_function(example, classes, class2id, tokenizer))

        # covariance calculations
        df = pd.DataFrame(tokenized_dataset["train"])
        new_columns_df = pd.DataFrame(df['labels'].tolist(), index=df.index, columns=classes)
        labels_array = new_columns_df.values
        covariance_matrix = np.cov(labels_array.T)
        correlation = new_columns_df.corr()

        # plot covariance
        correlation_plotting(classes, correlation, dataset_name)
        # covariance_plotting(classes, covariance_matrix, dataset_name)


def covariance_plotting(classes, covariance_matrix, dataset_name):
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    sns.heatmap(covariance_matrix,
                annot=True,  # Show the covariance values on the heatmap
                fmt='.2f',  # Format the annotation values to one decimal place
                annot_kws={"size": 2 if len(covariance_matrix) > 20 else 8},
                cmap='RdBu',
                center=0,
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Covariance Matrix Heatmap')
    # open output
    fpath = "llm-goal-scope/data/qa_dataset/results/"
    dataset_name = dataset_name.split(".")[0]
    dataset_name = dataset_name.split("/")[2:]
    dataset_name = "_".join(dataset_name)
    os.makedirs("llm-goal-scope/data/qa_dataset/results/" + dataset_name + "/", exist_ok=True)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, va='top')
    plt.tick_params(axis='both', which='major', labelsize=2 if len(covariance_matrix) > 20 else 8)
    plt.tight_layout()
    plt.savefig(fpath + dataset_name + "/covariance.png", dpi=300)
    plt.show()


def correlation_plotting(classes, correlation_matrix, dataset_name):
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    sns.heatmap(correlation_matrix,
                annot=True,  # Show the covariance values on the heatmap
                annot_kws={"size": 2 if len(correlation_matrix) > 20 else 8},
                fmt='.1f',  # Format the annotation values to one decimal place
                cmap='RdBu',
                center=0,
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Correlation Matrix Heatmap')
    # open output
    fpath = "./data/dataset/results/"
    dataset_name = dataset_name.split(".")[1]
    dataset_name = dataset_name.split("/")[3:]
    dataset_name = "_".join(dataset_name)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, va='top')
    plt.tick_params(axis='both', which='major', labelsize=2 if len(correlation_matrix) > 20 else 8)
    plt.tight_layout()
    os.makedirs(fpath + dataset_name + "/", exist_ok=True)
    plt.savefig(fpath + dataset_name + "/correlation.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
