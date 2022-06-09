import glob
import gzip
import os
import json

from tqdm import tqdm


DATA_DIR = "/mnt/datasets/c4/en"


def main():
    # combine all train files in a single .json file
    train_files = glob.glob(os.path.join(DATA_DIR, "c4-train*.json.gz"))
    train_files.sort()
    train_files = train_files[:10] # for now let's use the first 1o files only

    # read individual json samples
    samples = []
    for file in tqdm(train_files, desc="reading train files"):
        with gzip.open(file, 'r') as fin:
            for line in fin:  # each file contains multiple json lines (dicts)
                data = json.loads(line.decode('utf-8'))
                samples.append(data)

    # write data to disc
    with open("/mnt/datasets/c4/en/c4-train.json", 'w', encoding="utf-8") as outfile:
        for sample in tqdm(samples, desc="writing samples"):
            json.dump(sample, outfile)
            outfile.write('\n')


if __name__ == "__main__":
    main()
