import json
import pandas as pd


def main(directory):
    df = pd.read_csv(directory + "input_data.csv")

    # reference output format
    # {"prompt": <prompt>, "referenceResponse": <answer>, "category": <category>}

    # add this string as a new column in pandas

    #•	Intended application of results
    # •	Limitations due to methodological choices
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
