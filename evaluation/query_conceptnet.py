import os
import csv

from evaluation import conceptnet_collection, dbname


def read_csv_file(filename):
    assert os.path.exists(filename), f"{filename} does not exist"

    with open(filename, "r") as filereader:
        next(filereader)  # skip first line
        for line in filereader:
            elems = line.strip().split(",")
            yield elems[0], elems[1]


def store_into_collection(filename):
    for c1, c2 in read_csv_file(filename):
        d = {"concept_1": c1, "concept_2": c2}
        conceptnet_collection.insert_one(d)


def query_concept_1(c1):
    elems = conceptnet_collection.find({"concept_1": c1})
    return [" ".join(e["concept_2"].split("_")) for e in elems]


if __name__ == "__main__":
    # store_into_collection('../conceptnet_extractor/related_concepts.csv')
    print(query_concept_1("4_2_4"))
