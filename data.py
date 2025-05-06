from datasets import load_dataset
from collections import Counter
# there is only a train set
ds = load_dataset("biglam/gutenberg-poetry-corpus")['train']
# does this train slice add indices (slow?)


# counts = {}
# for line in ds:
#     id = line['gutenberg_id']
#     counts[id] = 1 + counts.get(id, 0)
    
# counts = Counter(counts)
# most_common = counts.most_common(10)
# avg = len(counts) / counts.total

lines = ds[::2]['line']
# for writing and trimming dataset
def write_lines(lines):
    with open("poetry.txt", "w") as f:
        f.writelines(lines)

# finetune using this notebook
# https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce
