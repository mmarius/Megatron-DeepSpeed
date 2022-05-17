import glob
import gzip
import os
import json

DATA_DIR = "/mnt/datasets/c4/en"
FILE = "/mnt/datasets/c4/en/c4-train.01023-of-01024.json.gz"

# samples = []
# with gzip.open(FILE, 'r') as fin:
#     for line in fin:
#         data = json.loads(line.decode('utf-8'))
#         # print(data.keys())
#         # print(data)
#         samples.append(data)

# print(len(samples))


# with open("/mnt/datasets/c4/en/c4-train.01023-of-01024_100samples.json", 'w', encoding="utf-8") as outfile:
#     for entry in samples[:100]:
#         json.dump(entry, outfile)
#         outfile.write('\n')

# get all files in a directory
files = glob.glob(os.path.join(DATA_DIR, "c4-train*.json.gz"))
files.sort()
print(len(files))
print(files[:2])
