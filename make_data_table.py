import os
import pandas as pd
import json


def main(directory):
    """
    Remove some of the unnecessary input data from HESTIA
    :param directory: input directory of HESTIA data
    :return: cleaned input.csv files for use in building the json.ld dataset
    """
    # get the directory
    directory_path = directory + "cleaned/"
    df = pd.DataFrame()

    # for each file in the directory, iterate through and add to big table
    for entry_name in os.listdir(directory_path):
        with open(directory_path + entry_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # normalize data (completeness)
            table_data = pd.json_normalize(data)

        df = pd.concat([df, table_data])

    # drop unnecessary columns
    if "recalculated" in directory:
        df = df.drop(columns=['systemBoundaryCompleteness.@type', 'systemBoundaryCompleteness.updatedVersion',
                              "IAproduct_properties.0.term.@type",
                              "IAproduct_properties.0.term.termType", "IAproduct_properties.0.term.@id",
                              "IAproduct_properties.0.@type",
                              "IAproduct_properties.1.term.@type", "IAproduct_properties.1.min", "IAproduct_properties.1.sd",
                              "IAproduct_properties.1.term.termType", "IAproduct_properties.1.term.@id",
                              "IAproduct_properties.2.term.@id",
                              "IAproduct_properties.1.@type", "IAproduct_properties", "IAproduct_properties.0.date",
                              "IAproduct_properties.2.@type",
                              "IAproduct_properties.0.min", "IAproduct_properties.0.max",
                              "IAproduct_properties.2.methodClassification",
                              "IAproduct_properties.2.methodClassificationDescription",
                              "IAproduct_properties.0.statsDefinition",
                              "IAproduct_properties.1.max", "IAproduct_properties.1.statsDefinition",
                              "IAproduct_properties.0.sd", "IAproduct_properties.1.methodClassification",
                              "IAproduct_properties.1.methodClassificationDescription",
                              "IAproduct_properties.0.methodClassification",
                              "IAproduct_properties.0.methodClassificationDescription", "IAproduct_properties.3.term.@type",
                              "IAproduct_properties.3.term.termType",
                              "IAproduct_properties.3.term.@id", "IAproduct_properties.3.@type",
                              "IAproduct_properties.0.methodModelDescription",
                              "IAproduct_properties.1.methodModelDescription", "IAproduct_properties.2.term.@type",
                              "IAproduct_properties.2.term.termType"
                              ])
    else:
        df = df.drop(columns=['systemBoundaryCompleteness.@type', "IAproduct_properties.0.term.@type",
                              "IAproduct_properties.0.term.termType", "IAproduct_properties.0.term.@id",
                              "IAproduct_properties.0.@type",
                              "IAproduct_properties.1.term.@type", "IAproduct_properties.1.min", "IAproduct_properties.1.sd",
                              "IAproduct_properties.1.term.termType", "IAproduct_properties.1.term.@id",
                              "IAproduct_properties.2.term.@id",
                              "IAproduct_properties.1.@type", "IAproduct_properties", "IAproduct_properties.0.date",
                              "IAproduct_properties.2.@type",
                              "IAproduct_properties.0.min", "IAproduct_properties.0.max",
                              "IAproduct_properties.2.methodClassification",
                              "IAproduct_properties.2.methodClassificationDescription",
                              "IAproduct_properties.0.statsDefinition",
                              "IAproduct_properties.1.max", "IAproduct_properties.1.statsDefinition",
                              "IAproduct_properties.0.sd", "IAproduct_properties.1.methodClassification",
                              "IAproduct_properties.1.methodClassificationDescription",
                              "IAproduct_properties.0.methodClassification",
                              "IAproduct_properties.0.methodClassificationDescription", "IAproduct_properties.3.term.@type",
                              "IAproduct_properties.3.term.termType",
                              "IAproduct_properties.3.term.@id", "IAproduct_properties.3.@type",
                              "IAproduct_properties.0.methodModelDescription",
                              "IAproduct_properties.1.methodModelDescription", "IAproduct_properties.2.term.@type",
                              "IAproduct_properties.2.term.termType"
                              ])

    # relabel columns - Hestia says FU are given by the term of the product (units)
    df.columns = df.columns.str.replace('units', 'functionalUnit')

    # write out file
    df.to_csv(directory + "input_data.csv", index=False)


if __name__ == "__main__":
    prefix = "./data/hestia/"  # "llm-goal-scope/data/hestia/" on AWS
    main(prefix)
    main(prefix + "recalculated/")
