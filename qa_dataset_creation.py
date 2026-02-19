import json
import pandas as pd
import itertools
import constants
from langchain_community.vectorstores import FAISS
import rag_retrieval
from tqdm import tqdm
import uuid
import os


def intendedApplication(row, RAG, vdb, reader, tokenizer):
    if len(row["intendedApplication"]) == 0:
        return ""
    else:
        question = "For this production system, what is the intended application of the LCA study?"
        if RAG:
            answer, docs = rag_retrieval.answer_with_rag(question, reader, tokenizer, vdb)
            context = " Additional Context: " + str(answer)
        else:
            context = ""
        return [{"labels": [row["intendedApplication"]],
                 "title": "Intended Application",
                 "id": str(uuid.uuid4()),
                 "context": row["systemDescription"] + context}]


def studyReasons(row, RAG, vdb, reader, tokenizer):
    if len(row["studyReasons"]) == 0:
        return ""
    else:
        question = "For this production system, what are the reasons for carrying out the LCA study?"
        if RAG:
            answer, docs = rag_retrieval.answer_with_rag(question, reader, tokenizer, vdb)
            context = " Additional Context: " + str(answer)
        else:
            context = ""
        return [{"labels": [row["studyReasons"]],
                 "title": "Study Reasons",
                 "id": str(uuid.uuid4()),
                 "context": row["systemDescription"] + context}]


def targetAudience(row, RAG, vdb, reader, tokenizer):
    if len(row["intendedAudience"]) == 0:
        return ""
    else:
        question = "For this production system, what is the target audience of the LCA study?"
        if RAG:
            answer, docs = rag_retrieval.answer_with_rag(question, reader, tokenizer, vdb)
            context = " Additional Context: " + str(answer)
        else:
            context = ""
        return [{"labels": [row["intendedAudience"]],
                 "title": "Target Audience",
                 "id": str(uuid.uuid4()),
                 "context": row["systemDescription"] + context}]


def comparativeAssertions(row, RAG, vdb, reader, tokenizer):
    if len(row["comparativeAssertions"]) == 0:
        return ""
    else:
        question = "For this production system, are these results to be used in comparative assertions?"
        if RAG:
            answer, docs = rag_retrieval.answer_with_rag(question, reader, tokenizer, vdb)
            context = " Additional Context: " + str(answer)
        else:
            context = ""
        return [{"labels": [row["comparativeAssertions"]],
                 "title": "Comparative Assertion",
                 "id": str(uuid.uuid4()),
                 "context": row["systemDescription"] + context}]


def actors(row, RAG, vdb, reader, tokenizer):
    question = "For this production system, who are the important actors?"
    if RAG:
        answer, docs = rag_retrieval.answer_with_rag(question, reader, tokenizer, vdb)
        context = " Additional Context: " + str(answer)
    else:
        context = ""
    if len(row["organization"]) == 0:
        return [
            {"labels": ["authors of the study", "authors and their collaborators"],
             "title": "Actors",
             "id": str(uuid.uuid4()),
             "context": row["systemDescription"] + context}]
    else:
        return [
            {"labels": [row["organization"], "authors of the study", "authors and their collaborators"],
             "title": "Actors",
             "id": str(uuid.uuid4()),
             "context": row["systemDescription"] + context}]


def product(row, RAG, vdb, reader, tokenizer):
    if len(row["name"]) == 0:
        return ""
    else:
        question = "For this production system, what product is the object of the assessment?"
        if RAG:
            answer, docs = rag_retrieval.answer_with_rag(question, reader, tokenizer, vdb)
            context = " Additional Context: " + str(answer)
        else:
            context = ""
        labels = [row["name"].split('-')[0].strip()]
        final_labels = []
        for j in labels:
            category = j.split(",")[0].strip()
            final_labels.append(j)
            final_labels.append(category)
        return [{"labels": final_labels,
                 "title": "Object of Assessment",
                 "id": str(uuid.uuid4()),
                 "context": row["systemDescription"] + context}]


def allocation(row, RAG, vdb, reader, tokenizer):
    if len(row["allocationMethod"]) == 0:
        return ""
    else:
        question = "For this production system, what is the appropriate allocation method?"
        if RAG:
            answer, docs = rag_retrieval.answer_with_rag(question, reader, tokenizer, vdb)
            context = " Additional Context: " + str(answer)
        else:
            context = ""
        return [{
            "labels": [row["allocationMethod"]],
            "title": "Allocation Method",
            "id": str(uuid.uuid4()),
            "context": row["systemDescription"] + context}]


def systemBoundary(row, RAG, vdb, reader, tokenizer):
    question = "What is included in the system boundary of this production system?"
    if RAG:
        answer, docs = rag_retrieval.answer_with_rag(question, reader, tokenizer, vdb)
        context = " Additional Context: " + str(answer)
    else:
        context = ""
    data = []
    labels = []
    # get all of the system boundary items and put them in the labels
    counter = 0
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
                    "id": str(uuid.uuid4()),
                    "context": row["systemDescription"] + " What is in the system boundary?" + context})
    return data


def functionalUnit(row, RAG, vdb, reader, tokenizer):
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
        question = "For this production system, what is the functional unit?"
        if RAG:
            answer, docs = rag_retrieval.answer_with_rag(question, reader, tokenizer, vdb)
            context = " Additional Context: " + str(answer)
        else:
            context = ""
        return [
            {"labels": fUnit,
             "title": "Functional Unit",
             "id": str(uuid.uuid4()),
             "context": row["systemDescription"] + context}]


def systemDescription(row):
    names = row["name"].split('-')
    if len(row["cycleDescription"]) > 0:
        return row["siteType"] + " producing " + names[0].strip() + " in " + names[
            1].strip() + ". Additional description: " + row["cycleDescription"] + "."
    return row["siteType"] + " producing " + names[0].strip() + " in " + names[1].strip() + "."

    
def process_all_tasks(row, RAG, vdb, reader, tokenizer):
    """Processes all columns for a single row in one go."""
    return pd.Series({
        "intendedApplicationQA": intendedApplication(row, RAG, vdb, reader, tokenizer),
        "studyReasonsQA": studyReasons(row, RAG, vdb, reader, tokenizer),
        "targetAudienceQA": targetAudience(row, RAG, vdb, reader, tokenizer),
        "comparativeAssertionsQA": comparativeAssertions(row, RAG, vdb, reader, tokenizer),
        "productQA": product(row, RAG, vdb, reader, tokenizer),
        "allocationQA": allocation(row, RAG, vdb, reader, tokenizer),
        "systemBoundaryQA": systemBoundary(row, RAG, vdb, reader, tokenizer),
        "functionalUnitQA": functionalUnit(row, RAG, vdb, reader, tokenizer)
    })
    
    
def main(output_directory, input_directory, RAG):
    tqdm.pandas()
    # read in data
    df = pd.read_csv(input_directory + "input_data.csv")
    # replace nan with empty strings
    df = df.fillna('')

    if RAG:
        embeddings = constants.EMBED_MODEL
        vdb = FAISS.load_local("llm-goal-scope/" + 
            constants.VDB_LOCATION, embeddings, allow_dangerous_deserialization=True)
    else:
        vdb = ""

    # set up llm models
    reader, tokenizer = rag_retrieval.model_config()

    # reference output format - add this string as a new column in pandas
    # [{"question": <prompt>, "labels": {'text': [<answer>], "answer_start": [0]}, "title": <category>, "context": <systemDescription>}, "id": <uuid>]

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
    tqdm.pandas(desc="Processing RAG Tasks")
    new_cols = df.progress_apply(lambda row: process_all_tasks(row, RAG, vdb, reader, tokenizer), axis=1)

    # Join the results back to your original dataframe
    df = pd.concat([df, new_cols], axis=1)

    # output the data
    print("\n append all questions to list")
    df = df[[col for col in df.columns if "QA" in col]]
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

        if RAG:
            fname = "rag_" + str(i) + ".jsonl"
        else:
            fname = str(i) + ".jsonl"

        with open(output_directory + fname, 'w') as f:
            for item in data:
                if item is not list:
                    json_line = json.dumps(item)
                else:
                    json_line = json.dumps(item[0])
                f.write(json_line + '\n')


if __name__ == "__main__":
    main("llm-goal-scope/data/qa_dataset/recalculated/rag/", "llm-goal-scope/data/hestia/recalculated/", True)
    main("llm-goal-scope/data/qa_dataset/original/rag/", "llm-goal-scope/data/hestia/", True)

    # DO NOT RERUN THESE WHEN UPDATING RAG FUNCTION - change uuid and is hard to track through git
    # main("llm-goal-scope/data/qa_dataset/recalculated/no_rag/", "llm-goal-scope/data/hestia/recalculated/",False)
    # main("llm-goal-scope/data/qa_dataset/original/no_rag/", "llm-goal-scope/data/hestia/",False)

