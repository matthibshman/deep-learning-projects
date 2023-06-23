import csv
import os

PATH = "counterfactual_data"
LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}
LABEL_INDEX = 2


def main():
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(PATH):
        for file in f:
            if ".tsv" in file:
                files.append(os.path.join(r, file))

    for f in files:
        with open(f) as file:
            tsv_file = csv.reader(file, delimiter="\t")

            data = []
            first = True
            for line in tsv_file:
                if first:
                    first = False
                    line[0] = "premise"
                    line[1] = "hypothesis"
                    line[2] = "label"
                else:
                    line[LABEL_INDEX] = LABEL_MAP[line[LABEL_INDEX]]

                data.append(line)

        with open(f.replace(".tsv", ".csv"), "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerows(data)


if __name__ == "__main__":
    main()
