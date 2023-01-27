import re
import pandas as pd

# Loading & processing train data
train = pd.DataFrame([
    {
        "Text": re.sub(r"\w+\:\w+", " ", text).strip(),
        "Category": re.search(r"(\w+)\:(\w+)", text).group(1).strip(),
    }
    for text in open("../data/train.txt", "r").readlines()
])[["Text", "Category"]]
print(f"train data loaded & processed...\n -shape: {train.shape}")

# Saving train data
train.to_csv("./train.csv", index=False)
print("train data saved...")

# loading & processing test data
test = pd.DataFrame([
    {
        "Text": re.sub(r"\w+\:\w+", " ", text).strip(),
        "Category": re.search(r"(\w+)\:(\w+)", text).group(1).strip(),
    }
    for text in open("../data/test.txt", "r").readlines()
])[["Text", "Category"]]
print(f"test data loaded & processed...\n -shape: {test.shape}")

# saving test data
test.to_csv("./test.csv", index=False)
print("test data saved...")
