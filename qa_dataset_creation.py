import json
import pandas as pd


def intendedApplication(row):
    if row["intendedApplication"].empty:
        return ""
    else:
        return {"prompt": "For this production system, " + row["systemDescription"] + ", what is the intended application of the LCA study?",
                "referenceResponse": [row["intendedApplication"]],
                "category": "Intended Application"}

def studyReasons(row):
    if row["studyReasons"].empty:
        return ""
    else:
        return {"prompt": "For this production system, " + row["systemDescription"] + ", what are the reasons for carrying out the LCA study?",
                "referenceResponse": [row["studyReasons"]],
                "category": "Study Reasons"}

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
    df["studyReasonsQA"] = df.apply(lambda row: studyReasons(row), axis=1)

    # •	Decision context and reasons for carrying out the study
    # •	Target audience
    # •	Comparative studies to be disclosed to the public
    # •	Commissioner of the study and other influential actors
    # •	Deliverables
    # •	Object of the assessment
    # •	LCI modelling framework and handling of multifunctional processes
    # •	System boundaries and completeness requirements
    # •	Representativeness of LCI data
    # •	Preparation of the basis for impact assessment
    # •	Special requirements for system comparisons
    # •	Needs for critical review
    # •	Planning reporting of results

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
