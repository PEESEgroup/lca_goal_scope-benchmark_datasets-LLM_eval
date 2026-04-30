import os
import json


def main(directory_path):
    """
    main method to clean data
    :param directory_path: path to location of HESTIA files
    :return: N/A
    """
    # for each file in the directory, iterate through
    for entry_name in os.listdir(directory_path+"ImpactAssessment/"):
        # get file path
        file_path = os.path.join(directory_path+"ImpactAssessment/", entry_name)
        extension = ".jsonld"
        # open file
        lca_data = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except UnicodeDecodeError:
                print("cannot read the file, skipping")
                continue  # can't read the file, so skipping

        # go through the list of available goal and scope tasks and find which ones are in Hestia
        # •	Intended application of results - in source
        # •	Limitations due to methodological choices - not available in current data
        # •	Decision context and reasons for carrying out the study - in source
        # •	Target audience - site
        if "site" in data:
            site = data['site']["@id"]
            try:
                with open(directory_path + "Site/{}".format(site)+extension, 'r', encoding='utf-8') as f_site:
                    try:
                        data_site = json.load(f_site)
                        if "description" in data_site:
                            lca_data['siteDescription'] = data_site["description"]
                        else:
                            lca_data['siteDescription'] = ""
                        if "name" in data_site:
                            lca_data['siteName'] = data_site["name"]
                        else:
                            lca_data['siteName'] = ""
                        if "siteType" in data_site:
                            lca_data['siteType'] = data_site["siteType"]
                        else:
                            lca_data['siteType'] = ""
                        """
                        # removed for now because hestia doesn't divulge this information
                        if "organisation" in data_site:
                            lca_data['organization'] = data_site["organisation"]
                        else:
                            lca_data['organization'] = ""
                        """
                        if "country" in data_site:
                            if "region" in data_site:
                                lca_data['siteLocation'] = data_site["country"]["name"] + " - " + data_site["region"]["name"]
                            else:
                                lca_data['siteLocation'] = data_site["country"]["name"]
                        else:
                            lca_data['siteLocation'] = ""
                    except UnicodeDecodeError as e:
                        print(e, entry_name, "skipping site")
                        lca_data['siteDescription'] = ""
                        lca_data['siteName'] = ""
                        lca_data['siteType'] = ""
                        lca_data['siteLocation'] = ""
                        # lca_data['organization'] = ""
            except FileNotFoundError as e:
                print(e, entry_name, "missing site")
                lca_data['siteDescription'] = ""
                lca_data['siteName'] = ""
                lca_data['siteType'] = ""
                lca_data['siteLocation'] = ""
                # lca_data['organization'] = ""

        # •	Comparative studies to be disclosed to the public - some studies have comparative studies,
        # which I think are called cycles
        cycle = data['cycle']["@id"]
        with open(directory_path + "Cycle/{}".format(cycle) + extension, 'r', encoding='utf-8') as f_cycle:
            try:
                data_cycle = json.load(f_cycle)
                if "description" in data_cycle:
                    lca_data['cycleDescription'] = data_cycle["description"]
                else:
                    lca_data['cycleDescription'] = ""
                if "functionalUnit" in data_cycle:
                    lca_data['functionalUnit'] = data_cycle["functionalUnit"]  #
                else:
                    lca_data['functionalUnit'] = ""
                # This FU is either ha or relative (meaning that the quantities of Inputs and Emissions correspond
                # to the quantities of Products) according to documentation
                if "completeness" in data_cycle:
                    lca_data["systemBoundaryCompleteness"] = data_cycle["completeness"]
                else:
                    lca_data['systemBoundaryCompleteness'] = ""
            except UnicodeDecodeError as e:
                print(e, entry_name, "skipping cycle")
                lca_data['cycleDescription'] = ""
                lca_data['systemBoundaryCompleteness'] = ""
                lca_data['functionalUnit'] = ""

        # •	Commissioner of the study and other influential actors - source
        source = data['source']["@id"]
        with open("llm-goal-scope/data/hestia/Source/{}".format(source) + extension, 'r', encoding='utf-8') as f_source:
            try:
                data_source = json.load(f_source)
                # while it would be nice to have a list of all the authors associated with the manuscript,
                # getting actors from hestia isn't readily apparent
                # it will be easiest to have the agent reply "authors of the study" or similar
                if "bibliography" in data_source:
                    if "documentDOI" in data_source["bibliography"]:
                        lca_data['DOI'] = data_source["bibliography"]["documentDOI"]
                    else:
                        lca_data['DOI'] = ""
                    if "title" in data_source["bibliography"]:
                        lca_data['title'] = data_source["bibliography"]["title"]
                    else:
                        lca_data['title'] = ""
                else:
                    lca_data['DOI'] = ""
                    lca_data['title'] = ""
                if "uploadNotes" in data_source:
                    lca_data['notes'] = data_source["uploadNotes"]
                else:
                    lca_data['notes'] = ""
                if "intendedApplication" in data_source:
                    lca_data['intendedApplication'] = data_source["intendedApplication"]
                else:
                    lca_data['intendedApplication'] = ""
                if "studyReasons" in data_source:
                    lca_data['studyReasons'] = data_source["studyReasons"]
                else:
                    lca_data['studyReasons'] = ""
                if "intendedAudience" in data_source:
                    lca_data['intendedAudience'] = data_source["intendedAudience"]
                else:
                    lca_data['intendedAudience'] = ""
                if "comparativeAssertions" in data_source:
                    lca_data['comparativeAssertions'] = data_source["comparativeAssertions"]
                else:
                    lca_data['comparativeAssertions'] = ""
            except UnicodeDecodeError as e:
                print(e, entry_name, "skipping source")
                lca_data['title'] = ""
                lca_data['DOI'] = ""
                lca_data['notes'] = ""
                lca_data['intendedApplication'] = ""
                lca_data['studyReasons'] = ""
                lca_data['intendedAudience'] = ""
                lca_data['comparativeAssertions'] = ""
        # •	Deliverables - not included in present data
        # •	Object of the assessment - name
        if "name" in data:
            lca_data['name'] = data['name']
        else:
            lca_data['name'] = ""
        # •	LCI modelling framework and handling of multifunctional processes - allocationMethod, LCI modelling
        # framework is recalculated to include all possible LCIA methods (107 midpoint methods at a count)
        if "allocationMethod" in data:
            lca_data['allocationMethod'] = data['allocationMethod']
        else:
            lca_data['allocationMethod'] = ""
        # •	System boundaries and completeness requirements - functional unit quantity, product, info in cycle.json
        if "functionalUnitQuantity" in data:
            lca_data['functionalUnitQuantity'] = data['functionalUnitQuantity']
        else:
            lca_data['functionalUnitQuantity'] = ""
        if "product" in data:
            if "fate" in data["product"]:
                lca_data["product_fate"] = data["product"]["fate"]
            else:
                lca_data['product_fate'] = ""
            if "properties" in data["product"]:
                # flattening handled when data table is made
                lca_data["product_properties"] = json.loads(json.dumps(dict(enumerate(data["product"]["properties"]))))
            else:
                lca_data['product_properties'] = ""
            if "primary" in data["product"]:
                lca_data["product_primary"] = data["product"]["primary"]
            else:
                lca_data['product_primary'] = ""
        else:
            lca_data['product_fate'] = ""
            lca_data['product_properties'] = ""
            lca_data['product_primary'] = ""
        # •	Representativeness of LCI data - not available
        # •	Preparation of the basis for impact assessment - not available in data
        # •	Special requirements for system comparisons - not available in data
        # •	Needs for critical review - not available in data
        # •	Planning reporting of results - not available in data, clearly results were reported

        # write out data
        with open(directory_path+"cleaned/{}".format(entry_name), "w+") as w:
            json.dump(lca_data, w, indent=4)


if __name__ == "__main__":
    main("llm-goal-scope/data/hestia/")
    main("llm-goal-scope/data/hestia/recalculated/")
