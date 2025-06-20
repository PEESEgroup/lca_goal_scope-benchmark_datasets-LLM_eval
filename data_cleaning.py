import os
import json

# get the directory
directory_path = "./data/recalculated/ImpactAssessment/"

# for each file in the directory, iterate through
for entry_name in os.listdir(directory_path):
    # get file path
    file_path = os.path.join(directory_path, entry_name)
    extension = ".jsonld"
    # open file and remove LCIA calculated values
    lca_data = {}
    with open(file_path, 'r') as f:
        data = json.load(f)

    # •	Intended application of results - not available in current data
    # •	Limitations due to methodological choices - not available in current data
    # •	Decision context and reasons for carrying out the study - not available in current data
    # •	Target audience - site
    site = data['site']["@id"]
    with open("./data/recalculated/Site/{}".format(site)+extension, 'r') as f_site:
        data_site = json.load(f_site)
        # TODO: clean site data
        lca_data['site'] = data_site
    # •	Comparative studies to be disclosed to the public - some studies have comparative studies, which I think are called cycles
    cycle = data['cycle']["@id"]
    with open("./data/recalculated/Cycle/{}".format(site) + extension, 'r') as f_cycle:
        data_cycle = json.load(f_cycle)
        # TODO: clean cycle data
        lca_data['cycle'] = data_cycle
    # •	Commissioner of the study and other influential actors - source
    source = data['source']["@id"]
    with open("./data/Source/{}".format(site) + extension, 'r') as f_source:
        data_source = json.load(f_source)
        # TODO: clean source data
        # TODO: get system boundary completeness data
        lca_data['source'] = data_source
    # •	Deliverables - not included in present data
    # •	Object of the assessment - name
    lca_data['name'] = data['name']
    # •	LCI modelling framework and handling of multifunctional processes - allocationMethod, LCI modelling framework is
    # recalculated to include all possible LCIA methods (107 midpoint methods at a count)
    lca_data['allocationMethod'] = data['allocationMethod']
    # •	System boundaries and completeness requirements - functional unit quantity, product, info in cycle.json
    lca_data['functionalUnitQuantity'] = data['functionalUnitQuantity']
    # TODO: clean product data
    lca_data["product"] = data["product"]
    # •	Representativeness of LCI data - not available, but recalculated by Hestia
    # •	Preparation of the basis for impact assessment - not available in data
    # •	Special requirements for system comparisons - not available in data
    # •	Needs for critical review - not available in data
    # •	Planning reporting of results - not available in data, clearly results were reported

    with open("data/cleaned/{}.json".format(entry_name), "w") as w:
        json.dump(lca_data, w)
