from datasets import load_dataset

data = load_dataset("IVN-RIN/BioBERT-Italian", split="train")

filtered = data.filter(lambda row: " bella " in row['text']) 
print(filtered)

# select the first two rows
sliced = filtered.select(range(2))
print(sliced)

# extract the 'text' for the first row
print(sliced[0]['text'])
