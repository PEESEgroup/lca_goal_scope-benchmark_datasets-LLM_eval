import json
import pandas as pd


def intendedApplication(row):
    if len(row["intendedApplication"])==0:
        return ""
    else:
        return {"prompt": "For this production system, " + row["systemDescription"] + ", what is the intended application of the LCA study?",
                "referenceResponse": [row["intendedApplication"]],
                "category": "Intended Application"}


def studyReasons(row):
    if len(row["studyReasons"])==0:
        return ""
    else:
        return {"prompt": "For this production system, " + row["systemDescription"] + ", what are the reasons for carrying out the LCA study?",
                "referenceResponse": [row["studyReasons"]],
                "category": "Study Reasons"}


def targetAudience(row):
    if len(row["intendedAudience"])==0:
        return ""
    else:
        return {"prompt": "For this production system, " + row["systemDescription"] + ", what is the target audience of the LCA study?",
                "referenceResponse": [row["intendedAudience"]],
                "category": "Target Audience"}


def comparativeAssertions(row):
    if len(row["comparativeAssertions"])==0:
        return ""
    else:
        return {"prompt": "For this production system, " + row["systemDescription"] + ", are these results to be used in comparative assertions?",
                "referenceResponse": [row["comparativeAssertions"]],
                "category": "Comparative Assertion"}

def main(directory):
    # read in data
    df = pd.read_csv(directory + "input_data.csv")
    # replace nan with empty strings
    df = df.fillna('')

    # reference output format - add this string as a new column in pandas
    # {"prompt": <prompt>, "referenceResponse": [<answer>], "category": <category>}

    # create a system description column
    df["systemDescription"] = df["siteType"] + " producing " + df["name"] + ". Additional description: " + df["cycleDescription"]

    #•	Intended application of results
    df["intendedApplicationQA"] =  df.apply(lambda row: intendedApplication(row), axis=1)

    # •	Limitations due to methodological choices - not available, skipping

    # •	Decision context and reasons for carrying out the study
    df["studyReasonsQA"] = df.apply(lambda row: studyReasons(row), axis=1)

    # •	Target audience
    df["targetAudienceQA"] = df.apply(lambda row: targetAudience(row), axis=1)

    # •	Comparative studies to be disclosed to the public
    df["comparativeAssertionsQA"] = df.apply(lambda row: comparativeAssertions(row), axis=1)

    # •	Commissioner of the study and other influential actors

    # •	Deliverables - not included, skipped
    # •	Object of the assessment

    # •	LCI modelling framework and handling of multifunctional processes - allocation here

    # •	System boundaries and completeness requirements - big boi

    # •	Representativeness of LCI data, not available, skipping
    # •	Preparation of the basis for impact assessment - LCIA method here [not in LCI Modelling framework]

    # •	Special requirements for system comparisons - not included, skipped
    # •	Needs for critical review -  not included, skipped
    # •	Planning reporting of results - not included, skipped

    # melt the data table and convert it into an array
    data = []
    output_filename = "qa_dataset.jsonl"

    # TODO: there is a limit of 1000 prompts per job in aws, so we might need to create multiple files

    # write out file
    with open(output_filename, 'w') as f:
        for item in data:
            json_line = json.dumps(item)
            f.write(json_line + '\n')



if __name__ == "__main__":
    main("./data/")
