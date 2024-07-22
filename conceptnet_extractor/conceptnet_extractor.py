import csv
import gzip

# Change this to the path of your downloaded ConceptNet assertions file
conceptnet_file_path = "/home/evilscript/Downloads/conceptnet-assertions-5.7.0.csv.gz"
output_csv_path = "usedfor_concepts.csv"

# Open the gzipped ConceptNet assertions file
with gzip.open(conceptnet_file_path, "rt", encoding="utf-8") as file:
    # Open the output CSV file
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["concept1", "concept2"])  # Write the CSV header

        # Read the ConceptNet assertions file line by line
        for line in file:
            parts = line.strip().split("\t")
            # Check if the line has the correct number of parts
            if len(parts) == 5:
                # Check if the relation is 'UsedFor'
                if parts[1] == "/r/UsedFor":
                    # Extract the start and end node labels
                    start_node = parts[2]
                    end_node = parts[3]
                    # Given that the nodes are stored like /c/en/cyberaddict/n we only want
                    # the lemma, i.e. the last part without the language and pos tag
                    concept1 = start_node.split("/")[3]
                    concept2 = end_node.split("/")[3]
                    # Optionally, you could load the JSON to get additional info, like weight
                    # json_info = json.loads(parts[4])
                    # weight = json_info['weight']
                    if (
                        start_node.split("/")[2] == "en"
                        and end_node.split("/")[2] == "en"
                    ):
                        csv_writer.writerow([concept1, concept2])

print(f"'UsedFor' relationships have been written to {output_csv_path}")
