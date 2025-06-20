import os
import json

# get the directory
directory_path = "./data/recalculated/ImpactAssessment/"

# for each file in the directory, iterate through
for entry_name in os.listdir(directory_path):
    # get file path
    file_path = os.path.join(directory_path, entry_name)
    extension = ".jsonld"
    # open file
    lca_data = {}
    with open(file_path, 'r') as f:
        data = json.load(f)

    # go through the list of available goal and scope tasks and find which ones are in Hestia
    # •	Intended application of results - not available in current data
    # •	Limitations due to methodological choices - not available in current data
    # •	Decision context and reasons for carrying out the study - not available in current data
    # •	Target audience - site
    site = data['site']["@id"]
    with open("./data/recalculated/Site/{}".format(site)+extension, 'r') as f_site:
        data_site = json.load(f_site)
        lca_data['siteDescription'] = data_site["description"]
        lca_data['siteName'] = data_site["name"]
        lca_data['siteType'] = data_site["siteType"]
        lca_data['siteLocation'] = data_site["country"]["name"] + " - " + data_site["region"]["name"]
    # •	Comparative studies to be disclosed to the public - some studies have comparative studies, which I think are called cycles
    cycle = data['cycle']["@id"]
    with open("./data/recalculated/Cycle/{}".format(cycle) + extension, 'r') as f_cycle:
        data_cycle = json.load(f_cycle)
        lca_data['cycleDescription'] = data_cycle["description"]
        lca_data['functionalUnit'] = data_cycle["functionalUnit"] #TODO: reconcile FU between here and data[product]
        lca_data["systemBoundaryCompleteness"] = data_cycle["completeness"]
    # •	Commissioner of the study and other influential actors - source
    source = data['source']["@id"]
    with open("./data/Source/{}".format(source) + extension, 'r') as f_source:
        data_source = json.load(f_source)
        lca_data['bibliography'] = data_source["bibliography"]
        lca_data['notes'] = data_source["uploadNotes"]
    # •	Deliverables - not included in present data
    # •	Object of the assessment - name
    lca_data['name'] = data['name']
    # •	LCI modelling framework and handling of multifunctional processes - allocationMethod, LCI modelling framework is
    # recalculated to include all possible LCIA methods (107 midpoint methods at a count)
    lca_data['allocationMethod'] = data['allocationMethod']
    # •	System boundaries and completeness requirements - functional unit quantity, product, info in cycle.json
    lca_data['functionalUnitQuantity'] = data['functionalUnitQuantity']
    lca_data["product_fate"] = data["product"]["fate"]
    for i in data["product"]["properties"]:
        lca_data["product{}properties".format(i)]["name"] = data["product"]["properties"][i]["term"]["name"]
        lca_data["product{}properties".format(i)]["units"] = data["product"]["properties"][i]["term"]["units"]
    lca_data["product_primary"] = data["product"]["primary"]
    # •	Representativeness of LCI data - not available, but recalculated by Hestia
    # •	Preparation of the basis for impact assessment - not available in data
    # •	Special requirements for system comparisons - not available in data
    # •	Needs for critical review - not available in data
    # •	Planning reporting of results - not available in data, clearly results were reported

    with open("data/cleaned/{}".format(entry_name), "w+") as w:
        json.dump(lca_data, w)
