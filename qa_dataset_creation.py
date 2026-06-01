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


def RAG_questions(dataset_type):
    """
    return the RAG questions
    :param dataset_type: the dataset type
    :return: relevant question
    """
    if dataset_type == "Allocation":
        return "What is the appropriate allocation method?"
    elif dataset_type == "Functional Unit":
        return "What is the functional unit?"
    elif dataset_type == "System Boundary":
        return "What is included in the system boundary of this production system?"
    elif dataset_type == "Product":
        return "What product is the object of the assessment?"
    else:
        return "Wrong Dataset Type"

    # questions for non existent datasets
    # intended_app = f"For the following production system, what is the intended application of the LCA study? Production system: {str(row['systemDescription'])}"
    # comparative = f"For the following production system, are these results to be used in comparative assertions? Production system: {str(row['systemDescription'])}"
    # target_audience = f"For the following production system, what is the target audience of the LCA study? Production system: {str(row['systemDescription'])}"
    # study_reason = f"For the following production system, what are the reasons for carrying out the LCA study? Production system: {str(row['systemDescription'])}"
    # actors = f"For the following production system, who are the important actors? Production system: {str(row['systemDescription'])}"

def HESTIA_information(dataset_type):
    """
    return the RAG questions
    :param dataset_type: the dataset type
    :return: relevant question
    """
    if dataset_type == "Allocation":
        return ("If system expansion is used, the available choices are either mass, economic, energy, or biophysical. "
                "If system expansion is not necessary, answer \"none required\". If system expansion does not need "
                "to be reported, answer \"none\".")
    elif dataset_type == "Functional Unit":
        return ("The functional unit can either be: \"1 ha\" (one hectare) or \"relative\" (meaning that the quantities "
                "of Inputs and Emissions correspond to the quantities of Products). If the primary product is a crop or "
                "forage, the functional unit must be 1 ha. If \"relative\" is reported above, please also provide the "
                "functional unit most relevant to the production system.")
    elif dataset_type == "System Boundary":
        return ("For each of the following categories, please report the system boundary completeness requirement for the life cycle assessment Cycle given in the description in the form '<category>: True/False'."
                "If the types and quantities of the category are specified in the life cycle assessment, set to True. If the category is not present in the life cycle assessment, set to True."
                "If the category was used, but the types and quantities are not specified, set to False. \n"
                "The categories include:\n"
                "animalFeed: The types and quantities of all animal feed used during the Cycle, including hay and silage. Note that fresh forage has its own completeness field.\n"
                "animalPopulation: The types and quantities of all live animals or live aquatic species that were present during the Cycle.\n"
                "cropResidue: The quantity of above and below ground crop residue created and its management are recorded.\n"
                "electricityFuel: The types and quantities of all electricity and fuel used during the Cycle, excluding during the transport phase.\n"
                "excreta: The types and quantities of excreta created and its management.\n"
                "fertiliser: The types and quantities of all organic fertiliser and inorganic fertiliser, or the quantity of each fertiliser brand name.\n"
                "freshForage: The types and quantities of all fresh forage fed to, or grazed by, animals during the Cycle.\n"
                "ingredient: For feed or food processing Cycles, the type and quantities of all feed or food ingredients used, such as crop products, animal products, processed foods, and/or feed or food additives.\n"
                "liveAnimalInput: The types and quantities of all live animals or live aquatic species which were Inputs into the Cycle. For example, piglets might be an Input into a pig fattening Cycle.\n"
                "material: The types and quantities of all material and substrate Inputs, which includes capital equipment depreciated over the Cycle.\n"
                "operation: The types of all mechanical operation performed during the Cycle and either their duration or the percentage of area they covered.\n"
                "otherChemical: The types and quantities of all other chemicals (including processing aids, other inorganic chemicals, and other organic chemicals) used during the Cycle.\n"
                "pesticideVeterinaryDrug: The types and quantities of all pesticides (either as active ingredients or brand names) and veterinary drugs used during the Cycle.\n"
                "seed: The types and quantities of all seed Inputs, such as seed,saplings, or semen.\n"
                "soilAmendment: The types and quantities of all soil amendments and biochar used during the Cycle.\n"
                "transport: The transport modes and distances for each Input to the Site are recorded. If Products were also Transported during this Cycle, the distances and modes are specified.\n"
                "waste: The types and quantities all waste streams, their and management, and their transport to where they are managed are specified (note that crop residue and excreta waste streams and management have their own completeness fields). Examples of waste streams include dead animals or plastic films for greenhouses. Examples of management include disposal into a water body or bio-digestion. Examples of transport include taking waste to a disposal center.\n"
                "water: The types and quantities of all water used during the Cycle.\n"
                "product: The types and quantities of all crop, live animal, live aquatic species, animal product, and processed food produced during the Cycle are recorded. In the case where Products were intended to be produced but no production occurred (e.g., if crops fail due to disease) the types of products should still be recorded and the quantity set to zero.")
    elif dataset_type == "Product":
        return "The Product produced during the production Cycle, which is the target of this Impact Assessment."
    else:
        return "Wrong Dataset Type"

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
        # open dataset and pass to RAG pipeline
        if ablation:
            input_data_files = ["./data/dataset/original/no_rag/Functional Unit.jsonl"]
            for f in tqdm(input_data_files):
                for j in range(8):
                    if j == 0:
                        ablation = "n-retrieved-10"
                    elif j == 1:
                        ablation = "n-retrieved-30"
                    elif j == 2:
                        ablation = "n-top-1"
                    elif j == 3:
                        ablation = "n-top-5"
                    elif j == 4:
                        ablation = "tokens-128"
                    elif j == 5:
                        ablation = "tokens-512"
                    elif j == 6:
                        ablation = "temp-033"
                    elif j == 7:
                        ablation = "temp-090"
                    out_fpath = "/".join(f.split("/")[:3]) + "/rag/" + f.split("/")[-1] + ablation
                    dataset_type = f.split("/")[-1]
                    with open(f, "r", encoding="utf-8") as infile, open(out_fpath, "w", encoding="utf-8") as outfile:
                        for line in tqdm(infile):
                            data_dict = json.loads(line)

                            # get relevant information
                            sys_descript = data_dict["context"]
                            question = RAG_questions(dataset_type)
                            hestia = HESTIA_information(dataset_type)
                            
                            # get RAG information back
                            if j == 0:
                                ablation = "n-retrieved-10"
                                answer, docs = rag_retrieval.answer_with_rag(sys_descript, question, hestia, reader, tokenizer, vdb, num_retrieved_docs=10)
                            elif j == 1:
                                ablation = "n-retrieved-30"
                                answer, docs = rag_retrieval.answer_with_rag(sys_descript, question, hestia, reader, tokenizer, vdb, num_retrieved_docs=30)
                            elif j == 2:
                                ablation = "n-top-1"
                                answer, docs = rag_retrieval.answer_with_rag(sys_descript, question, hestia, reader, tokenizer, vdb, num_docs_final=1)
                            elif j == 3:
                                ablation = "n-top-5"
                                answer, docs = rag_retrieval.answer_with_rag(sys_descript, question, hestia, reader, tokenizer, vdb, num_docs_final=1)
                            elif j == 4:
                                ablation = "tokens-128"
                                answer, docs = rag_retrieval.answer_with_rag(sys_descript, question, hestia, reader, tokenizer, vdb, num_tokens= 128)
                            elif j == 5:
                                ablation = "tokens-512"
                                answer, docs = rag_retrieval.answer_with_rag(sys_descript, question, hestia, reader, tokenizer, vdb, num_tokens=512)
                            elif j == 6:
                                ablation = "temp-033"
                                answer, docs = rag_retrieval.answer_with_rag(sys_descript, question, hestia, reader, tokenizer, vdb, temperature=0.3)
                            elif j == 7:
                                ablation = "temp-090"
                                answer, docs = rag_retrieval.answer_with_rag(sys_descript, question, hestia, reader, tokenizer, vdb, temperature=0.90)
                            context = " Additional Context: " + str(answer)
                            data_dict['context'] = sys_descript + context

                            # write back out to file
                            outfile.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
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
                dataset_type = f.split("/")[-1]
                with open(f, "r", encoding="utf-8") as infile, open(out_fpath, "w", encoding="utf-8") as outfile:
                    for line in tqdm(infile):
                        data_dict = json.loads(line)

                        # get relevant information
                        sys_descript = data_dict["context"]
                        question = RAG_questions(dataset_type)
                        hestia = HESTIA_information(dataset_type)
                        
                        # get RAG information back
                        answer, docs = rag_retrieval.answer_with_rag(sys_descript, question, hestia, reader, tokenizer, vdb)
                        context = " Additional Context: " + str(answer)
                        data_dict['context'] = sys_descript + context

                        # write back out to file
                        outfile.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
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
    main("", "", True, ablation=True)
    main("", "", True)
