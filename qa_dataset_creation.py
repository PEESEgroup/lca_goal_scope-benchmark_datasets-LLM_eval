import json
import pandas as pd
import re
import random
import itertools
import constants
from langchain_community.vectorstores import FAISS
import rag_retrieval
from tqdm import tqdm


def intendedApplication(row):
    if len(row["intendedApplication"]) == 0:
        return ""
    else:
        return [{"question": "For this production system, what is the intended application of the LCA study?",
                 "referenceResponse": [row["intendedApplication"]],
                 "id": "Intended Application",
                 "context": row["systemDescription"]}]


def studyReasons(row):
    if len(row["studyReasons"]) == 0:
        return ""
    else:
        return [{"question": "For this production system, what are the reasons for carrying out the LCA study?",
                 "referenceResponse": [row["studyReasons"]],
                 "id": "Study Reasons",
                 "context": row["systemDescription"]}]


def targetAudience(row):
    if len(row["intendedAudience"]) == 0:
        return ""
    else:
        return [{"question": "For this production system, what is the target audience of the LCA study?",
                 "referenceResponse": [row["intendedAudience"]],
                 "id": "Target Audience",
                 "context": row["systemDescription"]}]


def comparativeAssertions(row):
    if len(row["comparativeAssertions"]) == 0:
        return ""
    else:
        return [{"question": "For this production system, are these results to be used in comparative assertions?",
                 "referenceResponse": [row["comparativeAssertions"]],
                 "id": "Comparative Assertion",
                 "context": row["systemDescription"]}]


def actors(row):
    if len(row["organization"]) == 0:
        return [
            {"question": "For this production system, who are the important actors?",
             "referenceResponse": ["authors of the study", "authors and their collaborators"],
             "id": "Actors",
             "context": row["systemDescription"]}]
    else:
        return [
            {"question": "For this production system, who are the important actors?",
             "referenceResponse": [row["organization"], "authors of the study", "authors and their collaborators"],
             "id": "Actors",
             "context": row["systemDescription"]}]


def product(row):
    if len(row["name"]) == 0:
        return ""
    else:
        return [{"question": "For this production system, what product is the object of the assessment?",
                 "referenceResponse": [row["name"].split('-')[0].strip()],
                 "id": "Object of Assessment",
                 "context": row["systemDescription"]}]


def allocation(row):
    if len(row["allocationMethod"]) == 0:
        return ""
    else:
        return [{"question": "For this production system, what is the appropriate allocation method? Possible answers are: economic, "
                                   "mass, energy, biophysical, none, none required, system expansion.",
                 "referenceResponse": [row["allocationMethod"]],
                 "id": "Allocation Method",
                 "context": row["systemDescription"]}]


def systemBoundary(row):
    data = []
    for i in row.index.to_list():
        if "systemBoundaryCompleteness" in i:
            # replace camel case part of string so that it can go in the question
            sb_part = i.split('.')[1]
            sb_part = re.sub(r'([A-Z])', r' \1', sb_part).strip().lower()

            # standard output of questions
            if len(str(row[str(i)])) == 0:
                return ""
            else:
                data.append({"question": "True or False. For this production system, does the system boundary contain " + sb_part + "? ",
                             "referenceResponse": [str(row[str(i)]).capitalize()],
                             "id": "System Boundary Completeness",
                             "context": row["systemDescription"]})
    return data


def functionalUnit(row):
    fUnit = []
    if len(row["functionalUnit"]) != 0:
        fUnit.append(row["functionalUnit"])
    if len(row["product_properties.0.term.functionalUnit"]) != 0:
        fUnit.append(row["product_properties.0.term.functionalUnit"])
        fraction = row["product_properties.0.term.functionalUnit"].split('/')
        if len(fraction) > 1:
            fUnit.append(fraction[1].strip())
            fUnit.append(fraction[0].strip() + "/" + fraction[1].strip())
        fUnit.append(fraction[0].strip())
    if len(row["product_properties.1.term.functionalUnit"]) != 0:
        fUnit.append(row["product_properties.1.term.functionalUnit"])
        fraction = row["product_properties.1.term.functionalUnit"].split('/')
        if len(fraction) > 1:
            fUnit.append(fraction[1].strip())
            fUnit.append(fraction[0].strip() + "/" + fraction[1].strip())
        fUnit.append(fraction[0].strip())
    if len(row["product_properties.2.term.functionalUnit"]) != 0:
        fUnit.append(row["product_properties.2.term.functionalUnit"])
        fraction = row["product_properties.2.term.functionalUnit"].split('/')
        if len(fraction) > 1:
            fUnit.append(fraction[1].strip())
            fUnit.append(fraction[0].strip() + "/" + fraction[1].strip())
        fUnit.append(fraction[0].strip())
    if len(row["product_properties.3.term.functionalUnit"]) != 0:
        fUnit.append(row["product_properties.3.term.functionalUnit"])
        fraction = row["product_properties.3.term.functionalUnit"].split('/')
        if len(fraction) > 1:
            fUnit.append(fraction[1].strip())
            fUnit.append(fraction[0].strip() + "/" + fraction[1].strip())
        fUnit.append(fraction[0].strip())

    if len(fUnit) == 0:
        return ""
    else:
        return [
            {"question": "For this production system, what is the functional unit?",
             "referenceResponse": fUnit,
             "id": "Functional Unit",
                "context": row["systemDescription"]}]


def systemDescription(row, RAG, knowledge_index):
    names = row["name"].split('-')

    # if RAG is enabled, add additional context to the description of the site
    if RAG:
        context = rag_retrieval.get_context(knowledge_index)
        if len(row["cycleDescription"]) > 0:
            return row["siteType"] + " producing " + names[0].strip() + " in " + names[
                1].strip() + ". Additional description: " + row["cycleDescription"] + ". Additional Context: " + context
        return row["siteType"] + " producing " + names[0].strip() + " in " + names[
            1].strip() + ". Additional Context: " + context
    else:
        if len(row["cycleDescription"]) > 0:
            return row["siteType"] + " producing " + names[0].strip() + " in " + names[
                1].strip() + ". Additional description: " + row["cycleDescription"]
        return row["siteType"] + " producing " + names[0].strip() + " in " + names[1].strip()


def main(directory, RAG):
    tqdm.pandas()
    # read in data
    df = pd.read_csv(directory + "input_data.csv")
    # replace nan with empty strings
    df = df.fillna('')

    if RAG:
        embeddings = constants.EMBED_MODEL
        vdb = FAISS.load_local(
            constants.VDB_LOCATION, embeddings, allow_dangerous_deserialization=True)

    # reference output format - add this string as a new column in pandas
    # [{"question": <prompt>, "referenceResponse": [<answer>], "id": <category>, "context": <systemDescription>}]

    # create a system description column
    df["systemDescription"] = df.progress_apply(lambda row: systemDescription(row, RAG, vdb), axis=1)

    #•	Intended application of results
    df["intendedApplicationQA"] = df.progress_apply(lambda row: intendedApplication(row), axis=1)

    # •	Limitations due to methodological choices - not available, skipping
    # •	Decision context and reasons for carrying out the study
    df["studyReasonsQA"] = df.progress_apply(lambda row: studyReasons(row), axis=1)

    # •	Target audience
    df["targetAudienceQA"] = df.progress_apply(lambda row: targetAudience(row), axis=1)

    # •	Comparative studies to be disclosed to the public
    df["comparativeAssertionsQA"] = df.progress_apply(lambda row: comparativeAssertions(row), axis=1)

    # •	Commissioner of the study and other influential actors - not currently included
    # cannot easily get hestia to divulge actors and organizations, which are relevant here
    # df["actorsQA"] = df.progress_apply(lambda row: actors(row), axis=1)

    # •	Deliverables - not included, skipped
    # •	Object of the assessment - excluding location and date
    df["productQA"] = df.progress_apply(lambda row: product(row),
                               axis=1)  # we would expect llms to excel at this one because this info is in the given context

    # •	LCI modelling framework and handling of multifunctional processes - allocation here
    df["allocationQA"] = df.progress_apply(lambda row: allocation(row), axis=1)

    # •	System boundaries and completeness requirements - big boi
    df["systemBoundaryQA"] = df.progress_apply(lambda row: systemBoundary(row), axis=1)

    # •	Representativeness of LCI data, not available, skipping
    # •	Preparation of the basis for impact assessment - LCIA method not included in base ImpactAssessment, too many versions in recalculated
    df["functionalUnitQA"] = df.progress_apply(lambda row: functionalUnit(row), axis=1)

    # •	Special requirements for system comparisons - not included, skipped
    # •	Needs for critical review -  not included, skipped
    # •	Planning reporting of results - not included, skipped

    # melt the data table and convert it into an array
    data = []

    # append all column values to list
    df = df[[col for col in df.columns if "QA" in col]]
    for i in tqdm(df.columns):
        data.append(df[str(i)].tolist())

    # unnest sublists and remove empty strings
    data = list(itertools.chain.from_iterable(data))
    data = [item for item in data if item != ""]

    # shuffle list for training/testing purposes
    random.seed(42)
    random.shuffle(data)

    if RAG:
        fname = "rag_qa_dataset.jsonl"
    else:
        fname = "qa_dataset.jsonl"

    with open(directory + fname, 'w') as f:
        for item in data:
            json_line = json.dumps(item[0])
            f.write(json_line + '\n')


if __name__ == "__main__":
    main("./data/", True)
