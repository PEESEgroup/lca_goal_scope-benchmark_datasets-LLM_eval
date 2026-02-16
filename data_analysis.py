import os
from pathlib import Path
import pandas as pd

def main():
    # build mAP output tables for each of the four categories
    map_tables("llm-goal-scope/data/results/")

    # plot number of labels versus precision for each of the four categories
    # label_precision("llm-goal-scope/data/results/")


def map_tables(fpath):
    # get all the results files
    root_directory = Path("./llm-goal-scope/data/qa_dataset/results")

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
    # rag_original = rag_original.reset_index(drop=True)
    # rag_original.columns = ['model', 'mAP', 'dataset']
    # rag_recalculated = rag_recalculated.reset_index(drop=True)
    # rag_recalculated.columns = ['model', 'mAP', 'dataset']

    original = original.pivot(index='dataset', columns='model', values='mAP')
    recalculated = recalculated.pivot(index='dataset', columns='model', values='mAP')
    # rag_original = rag_original.pivot(index='dataset', columns='model', values='mAP')
    # rag_recalculated = rag_recalculated.pivot(index='dataset', columns='model', values='mAP')
    
    # print out dataframes
    original.to_csv("./llm-goal-scope/data/qa_dataset/results/mAP_original.csv")
    recalculated.to_csv("./llm-goal-scope/data/qa_dataset/results/mAP_recalculated.csv")
    # rag_original.to_csv("./llm-goal-scope/data/qa_dataset/results/mAP_rag_original.csv")
    # rag_recalculated.to_csv("./llm-goal-scope/data/qa_dataset/results/mAP_rag_recalculated.csv")



if __name__ == "__main__":
    main()