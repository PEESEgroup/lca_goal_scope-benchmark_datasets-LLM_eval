import os
import json

# get the directory
directory_path = "./data/recalculated/ImpactAssessment/"

# for each file in the directory, iterate through
for entry_name in os.listdir(directory_path):
    # get file path
    file_path = os.path.join(directory_path, entry_name)
    # open file and remove LCIA calculated values
    lca_data = {}
    with open(file_path, 'r') as f:
        data = json.load(f)

    # •	Intended application of results - not available in current data
    # •	Limitations due to methodological choices - not available in current data
    # •	Decision context and reasons for carrying out the study - not available in current data
    # •	Target audience - site
    #TODO: move info from site .json file to this one
    lca_data['site'] = data['site']
    # •	Comparative studies to be disclosed to the public - some studies have comparative studies, which I think are called cycles
    # •	Commissioner of the study and other influential actors - source
    # •	Deliverables - not included in present data
    # •	Object of the assessment - name
    # •	LCI modelling framework and handling of multifunctional processes - impacts, endpoints, allocationMethod
    # •	System boundaries and completeness requirements - functional unit quantity, product
    # •	Representativeness of LCI data - not available, but recalculated by Hestia
    # •	Preparation of the basis for impact assessment
    # •	Special requirements for system comparisons - not available in data
    # •	Needs for critical review - not available in data
    # •	Planning reporting of results - not available in data, clearly results were reported

    with open("data/cleaned/{}.json".format(entry_name), "w") as w:
        json.dump(lca_data, w)
