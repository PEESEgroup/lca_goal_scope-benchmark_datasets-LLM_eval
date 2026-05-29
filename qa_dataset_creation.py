import json
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
                 "source": "",
                 "study": "",
                 "cycle": "",
                 "site": "",
                 "default method classification": "",
                 "bibliography": ""}]


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
                 "source": "",
                 "study": "",
                 "cycle": "",
                 "site": "",
                 "default method classification": "",
                 "bibliography": ""}]


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
                 "source": "",
                 "study": "",
                 "cycle": "",
                 "site": "",
                 "default method classification": "",
                 "bibliography": ""}]


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
                 "source": "",
                 "study": "",
                 "cycle": "",
                 "site": "",
                 "default method classification": "",
                 "bibliography": ""}]


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
                 "source": "",
                 "study": "",
                 "cycle": "",
                 "site": "",
                 "default method classification": "",
                 "bibliography": ""}]


def product(row):
    """
    generate the dataset for the product of interest in LCA
    :param row: row of data
    :return: entry for json-ld dataset
    """
    if len(row["name"]) == 0:
        return ""
    else:
        labels = [row["name"].split('-')[0].strip()]
        final_labels = []
        for j in labels:
            category = j.split(",")[0].strip()
            final_labels.append(j)
            final_labels.append(category)
        return [{"labels": final_labels,
                 "title": "Object of Assessment",
                 "context": row["systemDescription"],
                 "source": "",
                 "study": "",
                 "cycle": "",
                 "site": "",
                 "default method classification": "",
                 "bibliography": ""}]


def allocation(row):
    """
    generate the dataset for the allocation method using in LCA
    :param row: row of data
    :return: entry for json-ld dataset
    """
    if len(row["allocationMethod"]) == 0:
        return ""
    else:
        return [{
            "labels": [row["allocationMethod"]],
            "title": "Allocation Method",
            "context": row["systemDescription"],
                 "source": "",
                 "study": "",
                 "cycle": "",
                 "site": "",
                 "default method classification": "",
                 "bibliography": ""}]


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
                 "source": "",
                 "study": "",
                 "cycle": "",
                 "site": "",
                 "default method classification": "",
                 "bibliography": ""})
    return data


def functionalUnit(row):
    """
    generate the dataset for the functional unit of an LCA product/process
    :param row: row of data
    :return: entry for json-ld dataset
    """
    fUnit = []
    if len(row["functionalUnit"]) != 0:
        fUnit.append(row["functionalUnit"])
    if len(row["product_properties.0.term.functionalUnit"]) != 0:
        fUnit.append(row["product_properties.0.term.functionalUnit"])
    if len(row["product_properties.1.term.functionalUnit"]) != 0:
        fUnit.append(row["product_properties.1.term.functionalUnit"])
    if len(row["product_properties.2.term.functionalUnit"]) != 0:
        fUnit.append(row["product_properties.2.term.functionalUnit"])
    if len(row["product_properties.3.term.functionalUnit"]) != 0:
        fUnit.append(row["product_properties.3.term.functionalUnit"])

    fUnit = [i.replace("/ ", "/").replace(" /", "/") for i in fUnit]
    fUnit = list(set(fUnit))  # remove duplicates

    if len(fUnit) == 0:
        return ""
    else:
        return [
            {"labels": fUnit,
             "title": "Functional Unit",
             "context": row["systemDescription"],
                 "source": "",
                 "study": "",
                 "cycle": "",
                 "site": "",
                 "default method classification": "",
                 "bibliography": ""}]


def systemDescription(row):
    """
    completes the system description for each LCA in HESTIA
    :param row: row of dataframe
    :return: sentence describing the LCA system
    """
    names = row["name"].split('-')
    if len(row["cycleDescription"]) > 0:
        return row["siteType"] + " producing " + names[0].strip() + " in " + names[
            1].strip() + ". Additional description: " + row["cycleDescription"] + "."
    return row["siteType"] + " producing " + names[0].strip() + " in " + names[1].strip() + "."


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


def main(output_directory, input_directory, RAG):
    """
    main method to convert input data into json-ld multi-label text classification dataset
    :param output_directory: output directory
    :param input_directory: input directory
    :param RAG: boolean to include RAG or not
    :return: N/A
    """
    tqdm.pandas()
    # read in data
    df = pd.read_csv(input_directory + "input_data.csv")
    # replace nan with empty strings
    df = df.fillna('')

    # if it is RAG, the deduplicated tables already exist, so much of data processing is not necessary

    if RAG:
        embeddings = constants.EMBED_MODEL
        vdb = FAISS.load_local("llm-goal-scope/" +
                               constants.VDB_LOCATION, embeddings, allow_dangerous_deserialization=True)

        # set up llm models
        reader, tokenizer = rag_retrieval.model_config()
        # TODO: open dataset and pass to RAG pipeline
    else:
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

        # system description needs to be created before other data
        tqdm.pandas(desc="Creating System Description")
        df["systemDescription"] = df.progress_apply(lambda row: systemDescription(row), axis=1)

        # further optimize code to create dataset
        tqdm.pandas(desc="Processing Datasets")
        new_cols = df.progress_apply(lambda row: process_all_tasks(row), axis=1)

        # Join the results back to your original dataframe
        df = pd.concat([df, new_cols], axis=1)

        # deduplicate dataframe
        subset_cols = []
        old_len = len(df)
        df.drop_duplicates(subset=subset_cols)
        print(f'removed duplicates{len(df)-old_len}')

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
    main("llm-goal-scope/data/qa_dataset/recalculated/rag/", "llm-goal-scope/data/hestia/recalculated/", False)
    main("llm-goal-scope/data/qa_dataset/original/rag/", "llm-goal-scope/data/hestia/", False)

    # DO NOT RERUN THESE WHEN UPDATING RAG FUNCTION
    # main("llm-goal-scope/data/qa_dataset/recalculated/no_rag/", "llm-goal-scope/data/hestia/recalculated/",False)
    # main("llm-goal-scope/data/qa_dataset/original/no_rag/", "llm-goal-scope/data/hestia/",False)
