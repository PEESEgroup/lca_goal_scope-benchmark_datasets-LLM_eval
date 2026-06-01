import json
import os

import pandas as pd
import itertools
import constants
from langchain_community.vectorstores import FAISS
import rag_retrieval
from tqdm import tqdm


def intendedApplication(row):
    """
    generate the dataset for the intended application of LCA
    :return: entry for json-ld dataset
    """
    if len(row["intendedApplication"]) == 0:
        return ""
    else:
        return [{"labels": [row["intendedApplication"]],
                 "title": "Intended Application",
                 "context": row["systemDescription"],
                 "source": row['sourceID'],
                 "DOI": row['DOI'],
                 "cycle": row['cycleID'],
                 "site": row['siteID'],
                 "default method classification": row['cycleMethodClassification'],
                 "original study title": row['title']}]


def studyReasons(row):
    """
    generate the dataset for the study reasons of LCA
    :param row: row of data
    :return: entry for json-ld dataset
    """
    if len(row["studyReasons"]) == 0:
        return ""
    else:
        return [{"labels": [row["studyReasons"]],
                 "title": "Study Reasons",
                 "context": row["systemDescription"],
                 "source": row['sourceID'],
                 "DOI": row['DOI'],
                 "cycle": row['cycleID'],
                 "site": row['siteID'],
                 "default method classification": row['cycleMethodClassification'],
                 "original study title": row['title']}]


def targetAudience(row):
    """
    generate the dataset for the target audience of LCA
    :param row: row of data
    :return: entry for json-ld dataset
    """
    if len(row["intendedAudience"]) == 0:
        return ""
    else:
        return [{"labels": [row["intendedAudience"]],
                 "title": "Target Audience",
                 "context": row["systemDescription"],
                 "source": row['sourceID'],
                 "DOI": row['DOI'],
                 "cycle": row['cycleID'],
                 "site": row['siteID'],
                 "default method classification": row['cycleMethodClassification'],
                 "original study title": row['title']}]


def comparativeAssertions(row):
    """
    generate the dataset for the comparative assertions being made of LCA
    :param row: row of data
    :return: entry for json-ld dataset
    """
    if len(row["comparativeAssertions"]) == 0:
        return ""
    else:
        question = f"For the following production system, are these results to be used in comparative assertions? Production system: {str(row['systemDescription'])}"
        return [{"labels": [row["comparativeAssertions"]],
                 "title": "Comparative Assertion",
                 "context": row["systemDescription"],
                 "source": row['sourceID'],
                 "DOI": row['DOI'],
                 "cycle": row['cycleID'],
                 "site": row['siteID'],
                 "default method classification": row['cycleMethodClassification'],
                 "original study title": row['title']}]


def actors(row):
    """
    generate the dataset for the actors relevant to LCA
    :param row: row of data
    :return: entry for json-ld dataset
    """
    if len(row["organization"]) == 0:
        return [
            {"labels": ["authors of the study", "authors and their collaborators"],
             "title": "Actors",
             "context": row["systemDescription"]}]
    else:
        return [
            {"labels": [row["organization"], "authors of the study", "authors and their collaborators"],
             "title": "Actors",
             "context": row["systemDescription"],
                 "source": row['sourceID'],
                 "DOI": row['DOI'],
                 "cycle": row['cycleID'],
                 "site": row['siteID'],
                 "default method classification": row['cycleMethodClassification'],
                 "original study title": row['title']}]


def product(row):
    """
    generate the dataset for the product of interest in LCA
    :param row: row of data
    :return: entry for json-ld dataset
    """
    if len(row["IA_productName"]) == 0:
        return ""
    else:
        label = row["IA_productName"]
        category = label.split(",")[0].strip()
        final_labels = list({label, category})
        return [{"labels": final_labels,
                 "title": "Product of Assessment",
                 "context": row["systemDescription"],
                 "source": row['sourceID'],
                 "DOI": row['DOI'],
                 "cycle": row['cycleID'],
                 "site": row['siteID'],
                 "default method classification": row['cycleMethodClassification'],
                 "original study title": row['title']}]


def allocation(row):
    """
    generate the dataset for the allocation method using in LCA
    :param row: row of data
    :return: entry for json-ld dataset
    """
    if len(row["IAallocationMethod"]) == 0:
        return ""
    else:
        return [{
            "labels": [row["IAallocationMethod"]],
            "title": "Allocation Method",
            "context": row["systemDescription"],
                 "source": row['sourceID'],
                 "DOI": row['DOI'],
                 "cycle": row['cycleID'],
                 "site": row['siteID'],
                 "default method classification": row['cycleMethodClassification'],
                 "original study title": row['title']}]


def systemBoundary(row):
    """
    generate the dataset for the system boundary completeness of LCA
    :param row: row of data
    :return: entry for json-ld dataset
    """
    data = []
    labels = []
    # get all of the system boundary items and put them in the labels
    for i in row.index.to_list():
        if "systemBoundaryCompleteness" in i:
            # standard output of questions
            if len(str(row[str(i)])) != 0:
                # by definition this is a true or false question
                # because this is a binary, we only need the true labels
                if str(row[str(i)]).capitalize() == "True":
                    # unique true label for each type of system boundary
                    labels.append(str(row[str(i)]).capitalize() + "_" + str(i).split(".")[1])

    # return the data object
    data.append({"labels": labels,
                 "title": "System Boundary Completeness",
                 "context": row["systemDescription"] + " What is in the system boundary?",
                 "source": row['sourceID'],
                 "DOI": row['DOI'],
                 "cycle": row['cycleID'],
                 "site": row['siteID'],
                 "default method classification": row['cycleMethodClassification'],
                 "original study title": row['title']})
    return data


def functionalUnit(row):
    """
    generate the dataset for the functional unit of an LCA product/process
    :param row: row of data
    :return: entry for json-ld dataset
    """
    fUnit = []
    if (len(row["IA_productUnit"]) == 0) or (len(row['functionalUnit']) == 0):
        return ""
    else:
        return [
            {"labels": [row["IA_productUnit"], row['functionalUnit']],
             "title": "Functional Unit",
             "context": row["systemDescription"],
                 "source": row['sourceID'],
                 "DOI": row['DOI'],
                 "cycle": row['cycleID'],
                 "site": row['siteID'],
                 "default method classification": row['cycleMethodClassification'],
                 "original study title": row['title']}]


def systemDescription(row):
    """
    completes the system description for each LCA in HESTIA
    :param row: row of dataframe
    :return: sentence describing the LCA system
    """
    names = row["IAname"].split('-')
    if len(row["cycleDescription"]) > 0:
        if len(row['siteType']) > 0:
            if len(row['siteDescription']) > 0:
                string_to_return = row["siteType"] + " producing " + names[0].strip() + " in " + names[
                    1].strip() + ". Cycle description: " + row["cycleDescription"] + ". Site description: " + row["siteDescription"]
            else:
                string_to_return = row["siteType"] + " producing " + names[0].strip() + " in " + names[
                    1].strip() + ". Cycle description: " + row["cycleDescription"] + "."
        else:
            if len(row['siteDescription']) > 0:
                string_to_return = names[0].strip() + " produced in " + names[
                    1].strip() + ". Cycle description: " + row["cycleDescription"] + ". Site description: " + row["siteDescription"]
            else:
                string_to_return = names[0].strip() + " produced in " + names[
                    1].strip() + ". Cycle description: " + row["cycleDescription"] + "."
    else:
        if len(row['siteType']) > 0:
            if len(row['siteDescription']) > 0:
                string_to_return = row["siteType"] + " producing " + names[0].strip() + " in " + names[1].strip() + ". Site description: " + row["siteDescription"]
            else:
                string_to_return = row["siteType"] + " producing " + names[0].strip() + " in " + names[1].strip() + "."
        else:
            if len(row['siteDescription']) > 0:
                string_to_return = names[0].strip() + " produced in " + names[1].strip() + ". Site description: " + row["siteDescription"]
            else:
                string_to_return = names[0].strip() + " produced in " + names[1].strip() + "."

    string_to_return = string_to_return.strip().replace("..", ".").capitalize()
    return string_to_return


def RAG_questions(row):
    fu = f"For the following production system, what is the functional unit? Production system: {str(row['systemDescription'])}"
    sb = f"What is included in the system boundary of this production system? Production system: {str(row['systemDescription'])}"
    alloc = f"For the following production system, what is the appropriate allocation method? If system expansion is used, the available choices are either mass, economic, energy, or biophysical. If system expansion is not necessary, answer \"none required\". If another allocation method is used, answer \"none\". Production system: {str(row['systemDescription'])}"
    intended_app = f"For the following production system, what is the intended application of the LCA study? Production system: {str(row['systemDescription'])}"
    comparative = f"For the following production system, are these results to be used in comparative assertions? Production system: {str(row['systemDescription'])}"
    target_audience = f"For the following production system, what is the target audience of the LCA study? Production system: {str(row['systemDescription'])}"
    study_reason = f"For the following production system, what are the reasons for carrying out the LCA study? Production system: {str(row['systemDescription'])}"
    product_of_a = f"For the following production system, what product is the object of the assessment? Production system: {str(row['systemDescription'])}"
    actors = f"For the following production system, who are the important actors? Production system: {str(row['systemDescription'])}"


def process_all_tasks(row):
    """
    driver function to create all datasets
    :param row: row of data
    :param RAG: boolean for whether or not RAG is implemented
    :param vdb: vector database
    :param reader: LLM pipeline
    :param tokenizer: LLM tokenizer
    :return: entry for json-ld dataset
    """
    return pd.Series({
        "Intended Application": intendedApplication(row),
        "Study Reasons": studyReasons(row),
        "Target Audience": targetAudience(row),
        "Comparative Assertions": comparativeAssertions(row),
        "Product": product(row),
        "Allocation": allocation(row),
        "System Boundary": systemBoundary(row),
        "Functional Unit": functionalUnit(row),
    })


def main(output_directory, input_directory, RAG, ablation=False):
    """
    main method to convert input data into json-ld multi-label text classification dataset
    :param output_directory: output directory
    :param input_directory: input directory
    :param RAG: boolean to include RAG or not
    :return: N/A
    """
    # if it is RAG, the deduplicated tables already exist, so much of data processing is not necessary
    if RAG:
        embeddings = constants.EMBED_MODEL
        vdb = FAISS.load_local("./" + constants.VDB_LOCATION, embeddings, allow_dangerous_deserialization=True)

        # set up llm models
        reader, tokenizer = rag_retrieval.model_config()
        # TODO: open dataset and pass to RAG pipeline
        if ablation:
            pass
        else:
            # Open the line-delimited JSON file safely
            input_data_files = ["./data/dataset/original/no_rag/Allocation.jsonl",
                                "./data/dataset/original/no_rag/Functional Unit.jsonl",
                                "./data/dataset/original/no_rag/Product.jsonl",
                                "./data/dataset/original/no_rag/System Boundary.jsonl",
                                "./data/dataset/standardized/no_rag/Functional Unit.jsonl",
                                "./data/dataset/standardized/no_rag/Product.jsonl",
                                "./data/dataset/standardized/no_rag/System Boundary.jsonl", ]
            for f in tqdm(input_data_files):
                out_fpath = "/".join(f.split("/")[:3]) + "/rag/" + f.split("/")[-1]
                with open(f, "r", encoding="utf-8") as file:
                    for line in tqdm(file):
                        # Parse the individual line string into a Python dict
                        data_dict = json.loads(line)

                        sys_descript = data_dict[""]

                        # Access your data fields
                        print(data_dict)
    else:
        tqdm.pandas()
        # read in data
        df = pd.read_csv(input_directory + "input_data.csv")
        # replace nan with empty strings
        df = df.fillna('')
        # List of goal and scope tasks
        # •	Intended application of results
        # •	Limitations due to methodological choices - not available, skipping
        # •	Decision context and reasons for carrying out the study
        # •	Target audience
        # •	Comparative studies to be disclosed to the public
        # •	Commissioner of the study and other influential actors - not currently included
        # cannot easily get hestia to divulge actors and organizations, which are relevant here
        # df["actorsQA"] = df.progress_apply(lambda row: actors(row), axis=1)
        # •	Deliverables - not included, skipped
        # •	Object of the assessment - excluding location and date
        # •	Special requirements for system comparisons - not included, skipped
        # •	Needs for critical review -  not included, skipped
        # •	Planning reporting of results - not included, skipped
        # •	LCI modelling framework and handling of multifunctional processes - allocation here
        # •	System boundaries and completeness requirements
        # •	Representativeness of LCI data, not available, skipping
        # •	Preparation of the basis for impact assessment - LCIA method not included in base ImpactAssessment, too many versions in recalculated

        # deduplicate by an a-priori knowledge of what columns are referenced throughout
        old_len = len(df)
        subset_columns = []
        for i in df.columns.to_list():
            if "systemBoundaryCompleteness" in i:
                subset_columns.append(i)
        subset_columns.extend(['IA_productName', "IAallocationMethod", "IA_productUnit", "IAname",
                               "cycleDescription", "siteType", "siteDescription"])
        df = df.drop_duplicates(subset=subset_columns)
        print(f'removed duplicates {old_len - len(df)}')

        # system description needs to be created before other data
        tqdm.pandas(desc="Creating System Description")
        df["systemDescription"] = df.progress_apply(lambda row: systemDescription(row), axis=1)

        # further optimize code to create dataset
        tqdm.pandas(desc="Processing Datasets")
        new_cols = df.progress_apply(lambda row: process_all_tasks(row), axis=1)

        # Join the results back to your original dataframe
        df = pd.concat([df, new_cols], axis=1)

        # deduplicate dataframe
        subset_cols = ['Product', 'Allocation', 'System Boundary', 'Functional Unit']
        # output the data
        df = df[subset_cols]
        for i in tqdm(df.columns):
            data = df[str(i)].tolist()

            # unnest sublists and remove empty strings
            data = list(itertools.chain.from_iterable(data))
            data = [item for item in data if item != ""]

            # get information on all the labels
            all_labels = []
            for item in data:
                all_labels.extend(item["labels"])

            all_labels = list(set(all_labels))
            # add all label information to the dataset
            for item in data:
                item["all_labels"] = "; ".join(all_labels)

            fname = str(i) + ".jsonl"

            with open(output_directory + fname, 'w') as f:
                for item in data:
                    if item is not list:
                        json_line = json.dumps(item)
                    else:
                        json_line = json.dumps(item[0])
                    f.write(json_line + '\n')


if __name__ == "__main__":
    prefix = "./data/hestia/"
    print(os.getcwd())
    #main("./data/dataset/standardized/no_rag/", prefix + "recalculated/", False)
    #main("./data/dataset/original/no_rag/", prefix, False)

    # because of how the methods were refactored, only need 1 call to make RAG datasets,
    # and another call to make ablation RAG datasets
    main("", "", True)
    main("", "", True, ablation=True)
