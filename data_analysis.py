import os
from pathlib import Path
import pandas as pd
from collections import Counter
from datasets import Dataset, load_dataset, DatasetDict
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    # build mAP output tables for each of the four categories
    # map_tables()

    # plot number of labels versus precision for each of the four categories
    # label_precision()
    
    # collate errors for each dataset
    collect_error_samples()
    
    
def collect_error_samples():
    root_directory = Path("./data/qa_dataset/results")

    # two dataframes for two different dataset types
    original = pd.DataFrame()
    recalculated = pd.DataFrame()

    # Use rglob to recursively find all files matching the pattern
    for file_path in root_directory.rglob('errors.csv'):
        rag = "" if "no" in str(file_path).split("_") else "rag"
        dataset_type = "original" if "original" in str(file_path).split("_") else "recalculated"

        # read in data and extract label precision, dataset name, and model name
        data = pd.read_csv(file_path, header=None)
        language_model = "/".join(str(file_path).split("/")[5:7])
        data["model"] = language_model
        dataset_name = str(file_path).split("/")[4].split("_")[-1]
        data["dataset"] = dataset_name
        data["rag"] = rag

        # assign data to appropriate dataframe
        if dataset_type == "original":
            original = pd.concat([original, data])
        elif dataset_type == "recalculated":
            recalculated = pd.concat([recalculated, data])
            
    # build histograms for frequency of sample errors
    # plot histograms
    for i in [original, recalculated]:
        counts = i[['sample_index', 'model', 'rag']].value_counts()

        counts.plot.hist(column="sample_index", stacked=True)
        plt.title("Histogram of Sample Data")
        plt.xlabel("Number of Models in Which a Sample is Labeled Incorrectly")
        plt.ylabel("Frequency")

        plt.savefig("llm-goal-scope/data/qa_dataset/results/" + str(dataset_type) +".png", dpi=300)
        plt.show()
        print("dataset precision plot saved")

    # save data
    original.to_csv("./data/qa_dataset/results/all_errors_original.csv")
    recalculated.to_csv("./data/qa_dataset/results/all_errors_recalculated.csv")

    # TODO: for those errors that are persistent (appear across multiple models) do a deeper analysis

    # TODO: deduplicate entries


def label_precision():
    root_directory = Path("./data/qa_dataset/results")

    # four dataframes for four different dataset types
    rag_original = pd.DataFrame()
    rag_recalculated = pd.DataFrame()
    original = pd.DataFrame()
    recalculated = pd.DataFrame()

    # Use rglob to recursively find all files matching the pattern
    for file_path in root_directory.rglob('test_metrics.csv'):
        rag = "" if "no" in str(file_path).split("_") else "_rag"
        dataset_type = "original" if "original" in str(file_path).split("_") else "recalculated"
        dataset_category = dataset_type + rag

        # read in data and extract label precision, dataset name, and model name
        data = pd.read_csv(file_path, header=None)
        language_model = "/".join(str(file_path).split("/")[5:7])
        data = data[data[0].str.contains("Average Precision for Label ")]
        data["label"] = data[0].str.split(" ").str[4:].str.join(" ")
        data["model"] = language_model
        dataset_name = str(file_path).split("/")[4].split("_")[-1]
        data["precision"] = data[1] # relabel map column
        data["dataset"] = dataset_name
        data["category"] = dataset_category
        data = data[["model", "label", "dataset", "precision", "category"]]  # clean columns

        # assign data to appropriate dataframe
        if dataset_category == "original":
            original = pd.concat([original, data])
        elif dataset_category == "recalculated":
            recalculated = pd.concat([recalculated, data])
        elif dataset_category == "original_rag":
            rag_original = pd.concat([rag_original, data])
        elif dataset_category == "recalculated_rag":
            rag_recalculated = pd.concat([rag_recalculated, data])
    
    # read in datasets and extract number of labels in the test set
    filenames = ["llm-goal-scope/data/qa_dataset/original/no_rag/systemBoundaryQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/original/no_rag/allocationQA.jsonl",  
                 "llm-goal-scope/data/qa_dataset/original/no_rag/functionalUnitQA.jsonl", 
                 "llm-goal-scope/data/qa_dataset/original/no_rag/productQA.jsonl", 
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/functionalUnitQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/productQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/no_rag/systemBoundaryQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/original/rag/rag_allocationQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/original/rag/rag_functionalUnitQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/original/rag/rag_productQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/original/rag/rag_systemBoundaryQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/rag/rag_functionalUnitQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/rag/rag_productQA.jsonl",
                 "llm-goal-scope/data/qa_dataset/recalculated/rag/rag_systemBoundaryQA.jsonl",
                 ]

    # for each dataset
    rag_original_test = pd.DataFrame()
    rag_recalculated_test = pd.DataFrame()
    original_test = pd.DataFrame()
    recalculated_test = pd.DataFrame()

    for k in filenames:
        # load the dataset
        dataset_rag = "" if "no_rag" in str(k).split("/") else "_rag"
        dataset_dataset_type = "original" if "original" in str(k).split("/") else "recalculated"
        dataset_dataset_category = dataset_dataset_type + dataset_rag
        dataset_name = k.split("/")[-1].split(".")[0]
        dataset = load_dataset('json', data_files=k) # shuffle dataset before splitting
        dataset = dataset.shuffle(seed=42)
        
        # 80% train, 20% test + validation
        train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=42)
        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
        test = test_valid['test']
        test_labels = test["labels"] # get the test labels
        
        # flatten and count the occurence of labels
        flattened_test_labels = list(itertools.chain.from_iterable(test_labels))
        counts = Counter(flattened_test_labels)
        counts = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])
        counts["dataset"] = dataset_name
        counts = counts.reset_index(names='label')

        # save the label information to the appropriate place
        if dataset_dataset_category == "original":
            original_test = pd.concat([original_test, counts])
        elif dataset_dataset_category == "recalculated":
            recalculated_test = pd.concat([recalculated_test, counts])
        elif dataset_dataset_category == "original_rag":
            rag_original_test = pd.concat([rag_original_test, counts])
        elif dataset_dataset_category == "recalculated_rag":
            rag_recalculated_test = pd.concat([rag_recalculated_test, counts])

    # merge datatables
    original = pd.merge(original, original_test, "left", ["dataset", "label"])
    recalculated = pd.merge(recalculated, recalculated_test, "left", ["dataset", "label"])
    rag_original = pd.merge(rag_original, rag_original_test, "left", ["dataset", "label"])
    rag_recalculated = pd.merge(rag_recalculated, rag_recalculated_test, "left", ["dataset", "label"])

    # plot scatterplot
    for i in [original, recalculated, rag_original, rag_recalculated]:
        for j in i["dataset"].unique():
            fig, ax = plt.subplots()
            df = i[i["dataset"] == j].copy(deep=True)
            df = df.dropna()
            df['precision'] = df['precision'].astype(float) # handle nan
            df['count'] = df['count'].astype(int)
            df = map_color(df, "model")
            print(df)
            for model in df["model"].unique():
                plotting_df = df[df["model"] == model]
                ax.scatter(x =plotting_df['count'], y=plotting_df['precision'], c=plotting_df["color"], label=model, alpha=0.7)
            plt.xlabel('Frequency of Label')
            plt.ylabel('Model Precision')
            plt.title('Sample size effect for dataset' + str(df["category"].unique()[0]) + str(j))
            plt.legend()
            plt.grid(True)
            plt.savefig("llm-goal-scope/data/qa_dataset/results/" + str(df["category"].unique()[0]) + str(j)+".png", dpi=300)
            plt.show()
            print("dataset precision plot saved")

    # save data
    original.to_csv("./data/qa_dataset/results/labels_original.csv")
    recalculated.to_csv("./data/qa_dataset/results/labels_recalculated.csv")
    rag_original.to_csv("./data/qa_dataset/results/labels_rag_original.csv")
    rag_recalculated.to_csv("./data/qa_dataset/results/labels_rag_recalculated.csv")


def map_color(df, col):
    color_d = dict(zip(df[col].unique(), sns.color_palette("hls", df[col].nunique())))
    df['color'] = df[col].map(color_d)
    return df


def map_tables():
    # get all the results files
    root_directory = Path("./data/qa_dataset/results")

    # four dataframes for four different dataset types
    rag_original = pd.DataFrame()
    rag_recalculated = pd.DataFrame()
    original = pd.DataFrame()
    recalculated = pd.DataFrame()

    # Use rglob to recursively find all files matching the pattern
    for file_path in root_directory.rglob('test_metrics.csv'):
        rag = "" if "no" in str(file_path).split("_") else "_rag"
        dataset_type = "original" if "original" in str(file_path).split("_") else "recalculated"
        dataset_category = dataset_type + rag

        # read in data and extract mean average precision, dataset name, and model name
        data = pd.read_csv(file_path, header=None)
        language_model = "/".join(str(file_path).split("/")[5:7])
        data = data[data[0] == "Mean Average Precision (mAP)"] 
        data["model"] = language_model
        dataset_name = str(file_path).split("/")[4].split("_")[-1]
        data["mAP"] = data[1] # relabel map column
        data["dataset"] = dataset_name
        data = data[["model", "mAP", "dataset"]]  # clean columns

        # assign data to appropriate dataframe
        if dataset_category == "original":
            original = pd.concat([original, data])
        elif dataset_category == "recalculated":
            recalculated = pd.concat([recalculated, data])
        elif dataset_category == "original_rag":
            rag_original = pd.concat([rag_original, data])
        elif dataset_category == "recalculated_rag":
            rag_recalculated = pd.concat([rag_recalculated, data])

    # pivot dataframes to be wide
    original = original.reset_index(drop=True)
    original.columns = ['model', 'mAP', 'dataset']
    recalculated = recalculated.reset_index(drop=True)
    recalculated.columns = ['model', 'mAP', 'dataset']
    rag_original = rag_original.reset_index(drop=True)
    rag_original.columns = ['model', 'mAP', 'dataset']
    rag_recalculated = rag_recalculated.reset_index(drop=True)
    rag_recalculated.columns = ['model', 'mAP', 'dataset']

    original = original.pivot(index='dataset', columns='model', values='mAP')
    recalculated = recalculated.pivot(index='dataset', columns='model', values='mAP')
    rag_original = rag_original.pivot(index='dataset', columns='model', values='mAP')
    rag_recalculated = rag_recalculated.pivot(index='dataset', columns='model', values='mAP')
    
    # print out dataframes
    original.to_csv("./data/qa_dataset/results/mAP_original.csv")
    recalculated.to_csv("./data/qa_dataset/results/mAP_recalculated.csv")
    rag_original.to_csv("./data/qa_dataset/results/mAP_rag_original.csv")
    rag_recalculated.to_csv("./data/qa_dataset/results/mAP_rag_recalculated.csv")


if __name__ == "__main__":
    main()