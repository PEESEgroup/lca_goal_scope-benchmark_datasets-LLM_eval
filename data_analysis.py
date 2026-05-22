import math
from pathlib import Path
import pandas as pd
from collections import Counter
from datasets import load_dataset
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import colorsys
import matplotlib.patches as mpatches
from collections import defaultdict
from sklearn.metrics import precision_score
import ast


def main():
    """
    A bunch of plotting functions for figures for manuscript and SI
    :return: N/A
    """
    # build SI figure on prediction threshold
    # prediction_threshold()

    # plot number of labels versus precision for each of the four categories
    # label_precision()
    # parameter_precision()

    # collate errors for each dataset based on RAG
    collect_rag_error_rates()

    # identify occurence of errors and the extent to which models and ground truths agree
    #inter_reviewer_alignment()

    # plot the frequency of error rates across 12 datasets
    plot_error_codes()


def get_label_precision(file_path):
    rag = "no rag" if "no" in str(file_path).split("_") else "rag"
    dataset_type = "original" if "original" in str(file_path).split("_") else "recalculated"
    language_model = "/".join(str(file_path).split("\\")[4:6])
    dataset_name = str(file_path).split("\\")[3].split("_")[-1]
    results_df = pd.DataFrame(index=[0])

    # read in data and extract label precision, dataset name, and model name
    data = pd.read_csv(file_path)
    ground_truth = np.array(data["true_labels"].apply(ast.literal_eval).tolist())
    predictions = np.array(data["preds_70"].apply(ast.literal_eval).tolist())

    # update df with parameters
    results_df["model"] = language_model
    results_df["dataset"] = dataset_name.replace("QA", "")
    results_df["dataset_type"] = dataset_type
    results_df["RAG"] = rag
    P_results = []
    index = data["classes"].apply(ast.literal_eval).tolist()[0]

    # calculate precision score and save it
    for j in range(len(ground_truth[0])):
        p = precision_score(ground_truth[:, j], predictions[:, j], zero_division=0)
        P_results.append(p)

    # aggregate data together at the end of the loop iteration
    P_series = pd.DataFrame(pd.Series(P_results, index=index)).transpose()
    results_df = pd.concat([results_df, P_series], axis=1)
    return results_df


def prediction_threshold():
    """
        Calculates the extent to which the prediction threshold affects mAP
        :return: N/A
        """
    root_directory = Path("./data/qa_dataset/results")
    mWP_df = pd.DataFrame()
    mWP_csv_df = pd.DataFrame()

    # Use rglob to recursively find all files matching the pattern
    for file_path in root_directory.rglob('predictions.csv'):
        rag = "no rag" if "no" in str(file_path).split("_") else "rag"
        dataset_type = "original" if "original" in str(file_path).split("_") else "recalculated"
        language_model = "/".join(str(file_path).split("\\")[4:6])
        dataset_name = str(file_path).split("\\")[3].split("_")[-1]
        results_df = pd.DataFrame(index=[0])

        # read in data and extract label precision, dataset name, and model name
        data = pd.read_csv(file_path)
        data["logits"] = data["logits"].apply(ast.literal_eval)  # convert to literal lists
        ground_truth = np.array(data["true_labels"].apply(ast.literal_eval).tolist())

        # update df with parameters
        results_df["model"] = language_model
        results_df["dataset"] = dataset_name.replace("QA", "")
        results_df["dataset_type"] = dataset_type
        results_df["RAG"] = rag
        # mAP_results_df = results_df.copy(deep=True)
        mWP_results_df = results_df.copy(deep=True)

        # calculate effects of precision threshold for the BERT model in no RAG
        P_results = []
        for i in range(0, 101):
            # calculate the predictions given a threshold
            preds = data['logits'].apply(
                lambda x: [int((1 / (1 + np.exp(-float(item)))) > (i/100)) for item in x])
            threshold_preds = np.array(preds.tolist())

            # calculate precision score and save it
            p = precision_score(ground_truth, threshold_preds, average="macro", zero_division=0)
            if i == 70:  # 70 is the threshold, so we save the mWP value to a df
                mWP_results_df["mWP"] = p
                data["preds_70"] = preds
                data.to_csv(file_path, index=False)

            # Calculate Mean Average Precision (mAP) for the threshold value
            P_results.append(p)

        # aggregate data together at the end of the loop iteration
        P_series = pd.DataFrame(pd.Series(P_results, index=range(0, 101))).transpose()
        results_df = pd.concat([results_df, P_series], axis=1)
        mWP_csv_df = pd.concat([mWP_csv_df, mWP_results_df])
        mWP_df = pd.concat([mWP_df, results_df])

    # make a plot, one for each language model
    for j in mWP_df["model"].unique():
        fig, ax = plt.subplots(figsize=(14, 8))
        df = mWP_df[mWP_df["model"] == j]
        dft = df.T

        # make the model name, dataset type, and RAG info part of the column names
        header_rows = dft.iloc[1:4] # exclude model name
        new_column_names = header_rows.apply(lambda x: '_'.join(x.astype(str)))
        dft.columns = new_column_names  # make column headers important identifying info
        dft = dft[4:]  # keep only the numbers to plot

        # plot the data
        for col in dft.columns:
            # get color
            col_colors = col.split("_")
            if col_colors[0] == "allocation":
                cat_color = 0
            elif col_colors[0] == "functionalUnit":
                cat_color = 1
            elif col_colors[0] == "product":
                cat_color = 2
            elif col_colors[0] == "systemBoundary":
                cat_color = 3
            if col_colors[1] == "original":
                dat_color = 0
            elif col_colors[1] == "recalculated":
                dat_color = 1
            if col_colors[2] == "no rag":
                rag_color = 0
            elif col_colors[2] == "rag":
                rag_color = 1
            color_num = 4*cat_color+2*dat_color+rag_color
            cmap = plt.get_cmap('tab20c')

            ax.scatter(x=[i/100 for i in dft.index], y=dft[col], c=cmap(color_num), label=f"{col}")

        plt.xlabel('Threshold')
        plt.ylabel('Macro-weighted Precision')
        plt.title(f'{j}')
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.savefig(f"./data/qa_dataset/results/threshold_sensitivity_{j.split('/')[1]}.png", dpi=300)
        plt.show()

    # write out datasets to .csv
    # mAP_df = mAP_df.pivot(index=["dataset", "dataset_type"], columns="model", values="mWP").reset_index()
    mWP_csv_df = mWP_csv_df.pivot(index=["dataset", "dataset_type", "RAG"], columns="model", values="mWP").reset_index()
    # mAP_df.to_csv(f"./data/qa_dataset/results/mAP-no-thresholds.csv",index=False)
    mWP_csv_df.to_csv(f"./data/qa_dataset/results/mWP.csv", index=False)


def plot_error_codes():
    """
    Plots manually coded discrepancies between AI labeling and human labels as pie charts
    :return: saved .png image
    """
    df = pd.read_excel("./data/qa_dataset/results/All_Discrepancies_Coded.xlsx")
    df["Rationale"] = df['Rationale'].astype('category')
    group_cols = ["Dataset", "Dataset Type", "RAG"]
    df_counts = df.groupby(group_cols + ["Rationale"]).size().reset_index(name='Count')

    # Create a consistent color map for all Codes
    unique_codes = df['Rationale'].cat.categories.tolist()
    base_hues = {
        "A": 0.0,  # Red
        "B": 0.08,  # Orange
        "C": 0.15,  # Yellow-Gold
        "D": 0.33,  # Green
        "E": 0.5,  # Cyan
        "F": 0.66,  # Blue
        "G": 0.75,  # Purple
        "H": 0.85  # Magenta/Pink
    }
    color_map = {}
    for code in sorted(unique_codes):
        major = str(code).split(".")[0]
        # Get all sub-codes for this major group to determine shade depth
        subs = [c for c in unique_codes if str(c).split(".")[0] == major]
        rank = subs.index(code)

        # Calculate Lightness: starts dark and gets lighter
        # 0.3 is quite dark, 0.7 is lighter
        lightness = 0.3 + (rank * (0.4 / max(len(subs), 1)))

        # Convert HLS to Hex
        rgb = colorsys.hls_to_rgb(base_hues[major], lightness, 0.7)
        hex_color = '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb)

        color_map[str(code)] = hex_color

    # Create a single label for the x-axis by joining the group columns
    df_counts['Group'] = df_counts[group_cols].astype(str).agg(' | '.join, axis=1)
    df_counts = df_counts[df_counts['Count'] > 0]  # keep rationales that occur more than 0 times

    # Pivot the data: Rows = Groups, Columns = Rationale, Values = Count
    pivot_df = df_counts.pivot(index='Group', columns='Rationale', values='Count').fillna(0)

    fig, ax = plt.subplots(figsize=(14, 8))
    bottom = None
    for rationale in pivot_df.columns:
        counts = pivot_df[rationale]
        color = color_map.get(str(rationale), "#CCCCCC")  # Fallback to grey if missing
        ax.bar(
            pivot_df.index,
            counts,
            bottom=bottom,
            label=rationale,
            color=color,
            edgecolor='white',
            linewidth=0.5
        )

        # Update the bottom for the next rationale layer
        if bottom is None:
            bottom = counts
        else:
            bottom += counts

    major_groups = defaultdict(list)
    # sort unique_codes to ensure the legend order is logical
    for code in sorted(unique_codes):
        major = str(code).split(".")[0]
        major_groups[major].append(code)

    # Create the legend handles in "column order"
    # Matplotlib fills columns top-to-bottom, then left-to-right.
    # To make each major category its own column, we find the max rows needed.
    max_rows = max(len(v) for v in major_groups.values())
    legend_elements = []

    # Sort major keys and iterate
    sorted_majors = sorted(major_groups.keys())
    for major in sorted_majors:
        # Add the items for this major group
        for code in major_groups[major]:
            label = str(code)
            patch = mpatches.Patch(color=color_map[str(code)], label=label)
            legend_elements.append(patch)

        # Fill empty space with "invisible" patches to force the next major group
        # into the next column if they have different lengths
        for _ in range(max_rows - len(major_groups[major])):
            legend_elements.append(mpatches.Patch(color='none', label=''))

    # Set ncol to the number of major categories
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=len(sorted_majors),
        title="Discrepancy Classifications",
        bbox_to_anchor=(0.32, 0.8),
        frameon=True
    )
    ax.set_ylabel('Count')
    ax.set_title('Discrepancy Rationales by Dataset Group')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("./data/qa_dataset/results/errors-code-plot.png", dpi=300)
    plt.show()


def explain_discrepancies(df):
    """
    Preprocess data to give a simple textual description of the discrepancy so that I don't have to manually hunt for which position in the list the discrepancy occurs
    :param df: df of discrepancies made by the ML model
    :return: .csv sheet containing explanations of error codes
    """
    counts = train_label_frequency()

    # for each error in the dataframe, prepare an explanation.
    discrepancy_lines = []

    for index, row in df.iterrows():
        # extract relevant labels and convert from string to list
        preds = row['predicted_labels'].replace("\"", "").replace("[", "").replace("]", "").replace("'", "").split(", ")
        trues = row['true_labels'].replace("\"", "").replace("[", "").replace("]", "").replace("'", "").split(", ")
        class_names = row['classes']
        class_names = class_names.split("',")
        class_names = [i.replace("\"", "").replace("[", "").replace("]", "").replace(" '", "").replace("'", "") for i in
                       class_names]

        # Iterate through the labels for the current row
        # Using zip to compare predictions and true labels side-by-side
        for i, (p, t) in enumerate(zip(preds, trues)):
            if p != t:
                # identify the human-readable labels
                label_name = class_names[i]

                # represent the mismatch
                a_val = label_name if p == str(1) else f"No {label_name}"
                b_val = label_name if t == str(1) else f"No {label_name}"

                # lookup the frequency of the ground_truth in the training dataset
                if len(counts[counts['label'] == label_name]) == 0:
                    freq = 0  # there's an off chance the label is not found in the training dataset
                else:
                    right_label = counts[counts['label'] == label_name]
                    right_dataset = right_label[right_label["category"] == row["dataset_type"]]
                    if len(right_label) == 1:
                        if row["dataset"] != "allocationQA":
                            freq = 0  # there's an off chance the label is not found in the training dataset
                            # but is in the other type of dataset (standardized/recalculated). This is, of course,
                            # always the case for allocation, so those pings are excluded
                        else:
                            freq = right_dataset["percentage"].values[0]
                    else:
                        freq = right_dataset["percentage"].values[0]

                # if the model never predicted a label of 1, include that information
                line = f"ML model predicted {a_val} but the humans predicted {b_val}."

                # save data to a pd Series
                data = [row["context_for_errors"], line, freq, row["sample_index"], row["dataset"], row["dataset_type"],
                        row["rag"]]
                labels = ["Context", "Sentence", "Frequency", "Sample Index", "Dataset", "Dataset Type", "RAG"]
                s = pd.Series(data, index=labels)
                discrepancy_lines.append(s.to_frame().T)

    # output results to .csv
    discrepancies = pd.concat(discrepancy_lines)
    discrepancies = discrepancies.sort_values(by=['Sample Index'], ascending=[True])
    return discrepancies


def train_label_frequency():
    """
    Identifies how the frequency of the training label affects mAP
    :return: .png graphs
    """
    # read in datasets and extract number of labels in the test set
    filenames = ["data/qa_dataset/original/no_rag/systemBoundaryQA.jsonl",
                 "data/qa_dataset/original/no_rag/allocationQA.jsonl",
                 "data/qa_dataset/original/no_rag/functionalUnitQA.jsonl",
                 "data/qa_dataset/original/no_rag/productQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/functionalUnitQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/productQA.jsonl",
                 "data/qa_dataset/recalculated/no_rag/systemBoundaryQA.jsonl",
                 ]  # rag and no_rag datasets will be the same
    df_list = []
    for k in filenames:
        # load the dataset
        dataset_rag = "" if "no_rag" in str(k).split("/") else "_rag"
        dataset_dataset_type = "original" if "original" in str(k).split("/") else "recalculated"
        dataset_dataset_category = dataset_dataset_type + dataset_rag
        dataset_name = k.split("/")[-1].split(".")[0]
        dataset = load_dataset('json', data_files=k)  # shuffle dataset before splitting
        dataset = dataset.shuffle(seed=42)

        # 80% train, 20% test + validation
        train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=42)
        # Split the 10% test + valid in half test, half valid
        train_valid = train_testvalid['train']['labels']

        # flatten and count the occurence of labels in the training dataset
        flattened_test_labels = list(itertools.chain.from_iterable(train_valid))
        counts = Counter(flattened_test_labels)
        counts = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])

        # add in more identifying information
        counts["dataset"] = dataset_name if "_" not in dataset_name else dataset_name.split("_")[1]
        counts = counts.reset_index(names='label')
        counts["percentage"] = 100 * counts["count"] / len(train_valid)
        counts["category"] = dataset_dataset_category
        df_list.append(counts)

    df_list = pd.concat(df_list)
    return df_list


def inter_reviewer_alignment():
    """
    Calculates the extent to which the AI models differ from human labels and how frequently these labels differ
    :return: .csv with the number of LCAs in which AI and humans agree, as well as the number of discrepancies
    """
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

                    # TODO: send this dataframe to a new method to 1) find the frequency the wrong labels appear in the dataset
                    # and 2) write a generic sentence describing the mistake made - i.e. machine did x when human did y
                    discrepancies = explain_discrepancies(analysis_df)
                    discrepancies.to_csv(f"./data/qa_dataset/results/discrepancies_{rag}_{dataset_type}.csv",
                                         index=False)
                    analysis_df = analysis_df.sort_values(by='sample_index')
                    analysis_df = analysis_df.reset_index()
                    analysis_df.to_csv(f"./data/qa_dataset/results/ensemble_errors_{rag}_{dataset_type}.csv",
                                       index=False)

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
    """
    Identify the extent to which error rates appear in RAG and non-RAG models
    :return: N/A
    """
    root_directory = Path("./data/qa_dataset/results")

    # two dataframes for two different dataset types
    original = pd.DataFrame()
    recalculated = pd.DataFrame()

    # Use rglob to recursively find all files matching the pattern
    for file_path in root_directory.rglob('predictions.csv'):
        df = get_label_precision(file_path)

        # assign data to appropriate dataframe if there is data
        if len(df) > 0:
            if df["dataset_type"].unique()[0] == "original":
                original = pd.concat([original, df])
            elif df["dataset_type"].unique()[0] == "recalculated":
                recalculated = pd.concat([recalculated, df])

    # save data
    original.to_csv("./data/qa_dataset/results/all_errors_original.csv")
    recalculated.to_csv("./data/qa_dataset/results/all_errors_recalculated.csv")

    # Identify incidence of all/persistent errors in RAG
    error_analysis = {}
    for df in [original, recalculated]:
        for models in [["ESGBERT/EnvironmentalBERT-base", "FacebookAI/roberta-large",
                        "climatebert/distilroberta-base-climate-f", "google-bert/bert-base-uncased",
                        "microsoft/deberta-v3-base", "microsoft/deberta-v3-large", "microsoft/deberta-v3-small"],
                       ["google-bert/bert-base-uncased", "microsoft/deberta-v3-large",
                        "ESGBERT/EnvironmentalBERT-base"]]:
            dataset_type = df["dataset_type"].unique()[0]

            # subset df by the occurence of models
            analysis_df = df[df["model"].isin(models)]

            if len(models) < 7:  # if we are doing an ensemble estimate, apply it only to the case with fewer models
                # keep errors if they appear in the majority of models
                analysis_df = analysis_df.groupby(['sample_index', 'dataset']).filter(
                    lambda x: len(x) >= math.ceil(len(models) / 2))
                print(analysis_df)

            # find the percentage of rows that are in only rag, only no rag, or both
            # a row is defined as a row number and a dataset
            presence = pd.crosstab([analysis_df['sample_index'], analysis_df['dataset']], analysis_df['rag']).gt(0)
            only_rag_count = ((presence['rag'] == True) & (presence['no rag'] == False)).sum()
            only_no_rag_count = ((presence['no rag'] == True) & (presence['rag'] == False)).sum()
            both_count = ((presence['rag'] == True) & (presence['no rag'] == True)).sum()
            total = len(presence)

            # write data out to series
            data = [len(models), f"{only_rag_count / total:.1%}", f"{only_no_rag_count / total:.1%}",
                    f"{both_count / total:.1%}"]
            index_labels = ["Number of Models", 'RAG only', 'No RAG only', 'Both']
            s = pd.Series(data, index=index_labels)
            error_analysis[dataset_type + str(len(models))] = s

    # save error statistics
    df = pd.DataFrame(error_analysis)
    df = df.reset_index()
    df.to_csv(f"./data/qa_dataset/results/error_location.csv", index=False)


def parameter_precision():
    """
    Identify how mAP varies with the number of parameters in the LLM (millions)
    :return: .png file with results
    """
    model_parameters = {"model": ["climatebert/distilroberta-base-climate-f",
                                  "ESGBERT/EnvironmentalBERT-base",
                                  "FacebookAI/roberta-large",
                                  "google-bert/bert-base-uncased",
                                  "microsoft/deberta-v3-base",
                                  "microsoft/deberta-v3-large",
                                  "microsoft/deberta-v3-small"],
                        "parameters": [82.4, 82.8, 304, 110, 86, 304, 44]}

    df = pd.read_csv(f"./data/qa_dataset/results/mWP.csv")
    df = pd.melt(df, id_vars=['dataset', 'dataset_type', 'RAG'], var_name='model', value_name='mWP')
    mp = pd.DataFrame(model_parameters)
    df = pd.merge(df, mp, "left", on="model")

    # plotting parameters vs mAP
    fig, ax = plt.subplots()
    df['mWP'] = df['mWP'].astype(float)  # handle nan
    df['parameters'] = df['parameters'].astype(int)
    df = map_color(df, "dataset")
    for dataset in df["dataset"].unique():
        plotting_df = df[df["dataset"] == dataset]
        x = plotting_df['parameters']
        y = plotting_df['mWP']
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
    """
    Identify how mAP varies with the frequency of labels in the training dataset
    :return: .png with results
    """
    root_directory = Path("./data/qa_dataset/results")

    # four dataframes for four different dataset types
    rag_original = pd.DataFrame()
    rag_recalculated = pd.DataFrame()
    original = pd.DataFrame()
    recalculated = pd.DataFrame()

    # Use rglob to recursively find all files matching the pattern
    for file_path in root_directory.rglob('predictions.csv'):
        df = get_label_precision(file_path)

        # make the df long
        df = pd.melt(df, id_vars=['model', 'dataset', 'dataset_type', 'RAG'], var_name='label', value_name='precision')

        # assign data to appropriate dataframe
        if df["dataset_type"].unique()[0] == "original":
            original = pd.concat([original, df])
        elif df["dataset_type"].unique()[0] == "recalculated":
            recalculated = pd.concat([recalculated, df])

    original_test, rag_original_test, rag_recalculated_test, recalculated_test = get_test_samples()
    original_test["RAG"] = "no rag"
    recalculated_test["RAG"] = "no rag"
    rag_original_test["RAG"] = "rag"
    rag_recalculated_test["RAG"] = "rag"
    original_test = pd.concat([original_test, rag_original_test])
    recalculated_test = pd.concat([rag_recalculated_test, rag_original_test])

    # merge datatables
    original = pd.merge(original, original_test, "left", ["dataset", "label", "RAG"])
    recalculated = pd.merge(recalculated, recalculated_test, "left", ["dataset", "label", "RAG"])
    df = pd.concat([original, recalculated])

    # plot scatterplot
    fig, ax = plt.subplots()
    df = df.dropna()
    df['precision'] = df['precision'].astype(float)  # handle nan
    df['count'] = df['count'].astype(int)
    df = map_color(df, "dataset")
    for dataset in df["dataset"].unique():
        plotting_df = df[df["dataset"] == dataset]
        x = plotting_df['count']
        y = plotting_df['precision']
        # plot scatter plot
        ax.scatter(x, y, c=plotting_df["color"], label=dataset, alpha=0.7)

    plt.xlabel('Frequency of Label')
    plt.ylabel('Precision')
    plt.title('Sample size effect for all datasets')
    plt.legend()
    plt.grid(True)
    plt.savefig("./data/qa_dataset/results/sample-size_mAP.png", dpi=300)
    plt.show()
    print("dataset precision plot saved")

    # save data
    original.to_csv("./data/qa_dataset/results/labels_original.csv")
    recalculated.to_csv("./data/qa_dataset/results/labels_recalculated.csv")
    rag_original.to_csv("./data/qa_dataset/results/labels_rag_original.csv")
    rag_recalculated.to_csv("./data/qa_dataset/results/labels_rag_recalculated.csv")


def get_test_samples():
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
        dataset_name = k.split("/")[-1].split(".")[0].replace("QA", "")
        dataset = load_dataset('json', data_files=k)  # shuffle dataset before splitting
        dataset = dataset.shuffle(seed=42)

        # 80% train, 20% test + validation
        train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=42)
        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
        train_valid = train_testvalid['train']['labels']
        test = test_valid['test']

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
    return original_test, rag_original_test, rag_recalculated_test, recalculated_test


def map_color(df, col):
    """
    helper function to map colors for plots
    :param df: dataframe of interest
    :param col: column of interest
    :return: dataframe with new column
    """
    color_d = dict(zip(df[col].unique(), sns.color_palette("hls", df[col].nunique())))
    df['color'] = df[col].map(color_d)
    return df


def map_tables():
    """
    Create .csv files summarizing mAP results
    :return: 4 .csv files
    """
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
        data["mAP"] = data[1]  # relabel map column
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
