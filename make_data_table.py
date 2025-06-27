import os
import pandas as pd

def main():
    # get the directory
    directory_path = "./data/recalculated/ImpactAssessment/"
    df = pd.DataFrame()

    # for each file in the directory, iterate through and add to big table
    for entry_name in os.listdir(directory_path):
        json_file = pd.read_json(entry_name)
        df = pd.concat([df, json_file])

    # write out file
    df.to_csv("data/input_data", index=False)


if __name__ == "__main__":
    main()