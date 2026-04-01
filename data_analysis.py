import math
import os
from pathlib import Path
import pandas as pd
from collections import Counter
from datasets import load_dataset
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors
import csv
import numpy as np


def main():
    # build mAP output tables for each of the four categories
    # map_tables()

    # plot number of labels versus precision for each of the four categories
    # label_precision()
    # parameter_precision()
    
    # collate errors for each dataset based on RAG
    # collect_rag_error_rates()

    # identify occurence of errors and the extent to which models and ground truths agree
    inter_reviewer_alignment()  # TODO: Krippendorf alpha or similar...
    

def inter_reviewer_alignment():
    root_directory = Path("./data/qa_dataset/results")

    # two dataframes for two different dataset types
    original = pd.DataFrame()
    recalculated = pd.DataFrame()

    # Use rglob to recursively find all files matching the pattern
    for file_path in root_directory.rglob('errors.csv'):
        rag = "no rag" if "no" in str(file_path).split("_") else "rag"
        dataset_type = "original" if "original" in str(file_path).split("_") else "recalculated"

        # read in data and extract label precision, dataset name, and model name
        data = pd.read_csv(file_path)
        language_model = "/".join(str(file_path).split("\\")[4:6])
        data["model"] = language_model
        dataset_name = str(file_path).split("\\")[3].split("_")[-1]
        data["dataset"] = dataset_name
        data["dataset_type"] = dataset_type
        data["rag"] = rag

        # assign data to appropriate dataframe if there is data
        if len(data) > 0:
            if dataset_type == "original":
                original = pd.concat([original, data])
            elif dataset_type == "recalculated":
                recalculated = pd.concat([recalculated, data])

    # Identify the percentage of samples of LCA that have 0, 1, 2+ errors
    error_analysis = {}
    for df in [original, recalculated]:
        for rag in ["rag", "no rag"]:
            rag_df = df[df["rag"] == rag]
            for models in [["ESGBERT/EnvironmentalBERT-base", "FacebookAI/roberta-large",
                            "climatebert/distilroberta-base-climate-f", "google-bert/bert-base-uncased",
                            "microsoft/deberta-v3-base", "microsoft/deberta-v3-large", "microsoft/deberta-v3-small"],
                           ["google-bert/bert-base-uncased", "microsoft/deberta-v3-large",
                            "ESGBERT/EnvironmentalBERT-base"]]:
                dataset_type = rag_df["dataset_type"].unique()[0]

                # treating the ensemble prediction as a single model (see below)
                if len(models) == 7:
                    num_models = len(models)
                else:
                    num_models = 1
                num_rag_datasets = 2

                # calculate the total number of available samples based on the number of models
                if dataset_type == "original":
                    total_samples = 104 * num_models * num_rag_datasets
                else:
                    total_samples = 99 * num_models * num_rag_datasets

                # subset df by the occurence of models
                analysis_df = rag_df[rag_df["model"].isin(models)]

                if len(models) < 7:  # if we are doing an ensemble estimate, apply it only to the case with fewer models
                    # keep errors if they appear in the majority of models
                    analysis_df = analysis_df.groupby(['sample_index', 'dataset']).filter(
                        lambda x: len(x) >= math.ceil(len(models) / 2))

                    # remove duplicates (ensemble is treated as 1 model, so look for identical sample indexes, datasets, and RAG)
                    analysis_df = analysis_df.drop_duplicates(subset=['sample_index', 'dataset'])
                    analysis_df["model"] = "ensemble"
                    analysis_df = analysis_df.sort_values(by='sample_index')
                    analysis_df = analysis_df.reset_index()
                    analysis_df.to_csv(f"./data/qa_dataset/results/ensemble_errors_{rag}_{dataset_type}.csv", index=False)

                # group by unique sample identifiers of the sample, the model, and whether or not it is rag
                error_counts = analysis_df.groupby(['sample_index', 'model'])['dataset'].nunique()

                # count how many samples have exactly 1 error, 2+ errors, or 0 errors
                s_1_error = (error_counts == 1).sum()
                s_2_plus_errors = (error_counts >= 2).sum()
                s_0_errors = total_samples - len(error_counts)

                # write data out to series
                data = [num_models, rag, f"{s_0_errors / total_samples:.1%}", f"{s_1_error / total_samples:.1%}",
                        f"{s_2_plus_errors / total_samples:.1%}"]
                index_labels = ["Number of Models", "RAG", '0 Errors', '1 Error', '2+ Errors']
                s = pd.Series(data, index=index_labels)
                error_analysis[dataset_type + str(rag) + str(len(models))] = s

    # save error statistics
    df = pd.DataFrame(error_analysis)
    df = df.reset_index()
    df.to_csv(f"./data/qa_dataset/results/num_correct_LCA.csv", index=False)


def collect_rag_error_rates():
    root_directory = Path("./data/qa_dataset/results")

    # two dataframes for two different dataset types
    original = pd.DataFrame()
    recalculated = pd.DataFrame()

    # Use rglob to recursively find all files matching the pattern
    for file_path in root_directory.rglob('errors.csv'):
        rag = "no rag" if "no" in str(file_path).split("_") else "rag"
        dataset_type = "original" if "original" in str(file_path).split("_") else "recalculated"

        # read in data and extract label precision, dataset name, and model name
        data = pd.read_csv(file_path)
        language_model = "/".join(str(file_path).split("\\")[4:6])
        data["model"] = language_model
        dataset_name = str(file_path).split("\\")[3].split("_")[-1]
        data["dataset"] = dataset_name
        data["dataset_type"] = dataset_type
        data["rag"] = rag

        # assign data to appropriate dataframe if there is data
        if len(data) > 0:
            if dataset_type == "original":
                original = pd.concat([original, data])
            elif dataset_type == "recalculated":
                recalculated = pd.concat([recalculated, data])
            
    # build histograms for frequency of sample errors
    # for i in [original, recalculated]:
    #     # get the first indication of the frequency of error
    #     counts = i[['sample_index', 'dataset', 'rag']].value_counts().reset_index()
    #
    #     # pivot the dataframe
    #     counts = counts.pivot(columns=["dataset", "rag"], values="count")
    #
    #     # get colors
    #     cmap = cm.get_cmap('tab20')
    #     colors_rgba = [cmap(j) for j in range(20)]
    #     colors_hex = [matplotlib.colors.to_hex(c) for c in colors_rgba]
    #
    #     # plot df
    #     counts.plot.hist(
    #         bins=7,
    #         stacked=True,
    #         color=colors_hex,
    #         title='Stacked Histogram by Category (Pivoted Data)'
    #     )
    #
    #     plt.title("Histogram of Sample Data")
    #     plt.xlabel("Number of Models in Which a Sample is Labeled Incorrectly")
    #     plt.ylabel("Frequency")
    #
    #     plt.savefig("data/qa_dataset/results/" + str(i["dataset_type"].unique()[0]) +".png", dpi=300)
    #     plt.show()
    #     print("dataset precision plot saved")

    # save data
    original.to_csv("./data/qa_dataset/results/all_errors_original.csv")
    recalculated.to_csv("./data/qa_dataset/results/all_errors_recalculated.csv")

    # Identify incidence of all/persistent errors in RAG
    error_analysis = {}
    for df in [original, recalculated]:
        for models in [["ESGBERT/EnvironmentalBERT-base", "FacebookAI/roberta-large",
                        "climatebert/distilroberta-base-climate-f", "google-bert/bert-base-uncased",
                        "microsoft/deberta-v3-base", "microsoft/deberta-v3-large", "microsoft/deberta-v3-small"],
                       ["google-bert/bert-base-uncased", "microsoft/deberta-v3-large", "ESGBERT/EnvironmentalBERT-base"]]:
            dataset_type = df["dataset_type"].unique()[0]

            # subset df by the occurence of models
            analysis_df = df[df["model"].isin(models)]

            if len(models) < 7:  # if we are doing an ensemble estimate, apply it only to the case with fewer models
                # keep errors if they appear in the majority of models
                analysis_df = analysis_df.groupby(['sample_index', 'dataset']).filter(lambda x: len(x) >= math.ceil(len(models)/2))
                print(analysis_df)

            # find the percentage of rows that are in only rag, only no rag, or both
            # a row is defined as a row number and a dataset
            presence = pd.crosstab([analysis_df['sample_index'], analysis_df['dataset']], analysis_df['rag']).gt(0)
            only_rag_count = ((presence['rag'] == True) & (presence['no rag'] == False)).sum()
            only_no_rag_count = ((presence['no rag'] == True) & (presence['rag'] == False)).sum()
            both_count = ((presence['rag'] == True) & (presence['no rag'] == True)).sum()
            total = len(presence)

            # write data out to series
            data = [len(models), f"{only_rag_count / total:.1%}", f"{only_no_rag_count / total:.1%}", f"{both_count / total:.1%}"]
            index_labels = ["Number of Models", 'RAG only', 'No RAG only', 'Both']
            s = pd.Series(data, index=index_labels)
            error_analysis[dataset_type + str(len(models))] = s

    # save error statistics
    df = pd.DataFrame(error_analysis)
    df = df.reset_index()
    df.to_csv(f"./data/qa_dataset/results/error_location.csv", index=False)


def parameter_precision():
    root_directory = Path("./data/qa_dataset/results")
    df_list = []

    model_parameters = {
        "climatebert/distilroberta-base-climate-f": 82.4,
        "ESGBERT/EnvironmentalBERT-base": 82.8,
        "FacebookAI/roberta-large": 304,
        "google-bert/bert-base-uncased": 110,
        "microsoft/deberta-v3-base": 86,
        "microsoft/deberta-v3-large": 304,
        "microsoft/deberta-v3-small": 44
    }

    # Use rglob to recursively find all files matching the pattern
    for file_path in root_directory.rglob('test_metrics.csv'):
        rag = "" if "no" in str(file_path).split("_") else "_rag"
        dataset_type = "original" if "original" in str(file_path).split("_") else "recalculated"
        dataset_category = dataset_type + rag

        # read in data and extract label precision, dataset name, and model name
        data = pd.read_csv(file_path, header=None)
        language_model = "/".join(str(file_path).split("\\")[4:6])
        print(language_model)
        data = data[data[0].str.contains("Mean Average Precision")]
        data["mAP"] = data[1]
        data["model"] = language_model
        data["parameters"] = model_parameters[language_model]
        dataset_name = str(file_path).split("\\")[3].split("_")[-1]
        data["dataset"] = dataset_name
        data["category"] = dataset_category
        data = data[["model", "dataset", "parameters", "mAP", "category"]]  # clean columns
        df_list.append(data)

    df = pd.concat(df_list)

    # plotting parameters vs mAP
    fig, ax = plt.subplots()
    df['mAP'] = df['mAP'].astype(float) # handle nan
    df['parameters'] = df['parameters'].astype(int)
    df = map_color(df, "dataset")
    for dataset in df["dataset"].unique():
        plotting_df = df[df["dataset"] == dataset]
        x = plotting_df['parameters']
        y = plotting_df['mAP']
        # plot scatter plot
        ax.scatter(x, y, c=plotting_df["color"], label=dataset.strip("QA"), alpha=0.7)

        # add best fit line
        sort_idx = np.argsort(x)
        x_sorted = x.iloc[sort_idx]
        lin_coeffs = np.polyfit(x, y, 1)
        lin_fn = np.poly1d(lin_coeffs)
        ax.plot(x_sorted, lin_fn(x_sorted), color=plotting_df["color"].unique()[0], linestyle='--',
                alpha=0.6, label=f'{dataset} best fit line')

    plt.xlabel('Number of Model Parameters (Million)')
    plt.ylabel('mean Average Precision')
    plt.title('Effect of Number of Parameters')
    plt.legend()
    plt.grid(True)
    plt.savefig("./data/qa_dataset/results/num_params.png", dpi=300)
    plt.show()
    print("dataset num parameters plot saved")


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
        language_model = "/".join(str(file_path).split("\\")[4:6])
        data = data[data[0].str.contains("Average Precision for Label ")]
        data["label"] = data[0].str.split(" ").str[4:].str.join(" ")
        data["model"] = language_model
        dataset_name = str(file_path).split("\\")[3].split("_")[-1]
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
    filenames = ["data/qa_dataset/original/no_rag/systemBoundaryQA.jsonl",
                 "data/qa_dataset/original/no_rag/allocationQA.jsonl",
                 "data/qa_dataset/original/no_rag/functionalUnitQA.jsonl",
                 "data/qa_dataset/original/no_rag/productQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/functionalUnitQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/productQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/systemBoundaryQA.jsonl",
                 "data/qa_dataset/original/rag/rag_allocationQA.jsonl",
                 "data/qa_dataset/original/rag/rag_functionalUnitQA.jsonl",
                 "data/qa_dataset/original/rag/rag_productQA.jsonl",
                 "data/qa_dataset/original/rag/rag_systemBoundaryQA.jsonl",
                 "data/qa_dataset/recalculated/rag/rag_functionalUnitQA.jsonl",
                 "data/qa_dataset/recalculated/rag/rag_productQA.jsonl",
                 "data/qa_dataset/recalculated/rag/rag_systemBoundaryQA.jsonl",
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
        train_valid = train_testvalid['train']['labels']
        test = test_valid['test']
        test_labels = test["labels"] # get the test labels
        
        # flatten and count the occurence of labels in the training dataset
        flattened_test_labels = list(itertools.chain.from_iterable(train_valid))
        counts = Counter(flattened_test_labels)
        counts = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])
        counts["dataset"] = dataset_name if "_" not in dataset_name else dataset_name.split("_")[1]
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
        fig, ax = plt.subplots()
        df = i.dropna().copy(deep=True)
        df['precision'] = df['precision'].astype(float) # handle nan
        df['count'] = df['count'].astype(int)
        df = map_color(df, "dataset")
        for dataset in df["dataset"].unique():
            plotting_df = df[df["dataset"] == dataset]
            x = plotting_df['count']
            y = plotting_df['precision']
            # plot scatter plot
            ax.scatter(x, y, c=plotting_df["color"], label=dataset.strip("QA"), alpha=0.7)

        plt.xlabel('Frequency of Label')
        plt.ylabel('mean Average Precision')
        plt.title('Sample size effect for dataset' + str(df["category"].unique()[0]))
        plt.legend()
        plt.grid(True)
        plt.savefig("./data/qa_dataset/results/" + str(df["category"].unique()[0])+ " " + str(df["dataset"].unique()[0])+".png", dpi=300)
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
        language_model = "/".join(str(file_path).split("\\")[4:6])
        data = data[data[0] == "Mean Average Precision (mAP)"] 
        data["model"] = language_model
        dataset_name = str(file_path).split("\\")[3].split("_")[-1]
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
