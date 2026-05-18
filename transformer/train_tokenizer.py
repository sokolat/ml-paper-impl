"""
Train BasicTokenizer on some data with PyTorch/CUDA
"""

import os
import time
import torch
from minbpe import BasicTokenizer
import argparse
import pandas as pd

wmt_de_en_df = pd.read_csv("./archive/wmt14_translate_de-en_train.csv", skiprows=1, lineterminator='\n')
de_corpus = ' '.join(wmt_de_en_df.iloc[:, 0])
en_corpus = ' '.join(wmt_de_en_df.iloc[:, 1])
corpus = de_corpus + " " + en_corpus
# open some text and train a vocab of 512 tokens
# text = open("tests/enron.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()

# construct the Tokenizer object and kick off verbose training
tokenizer = BasicTokenizer()
tokenizer.train(corpus, 37000, verbose=True)
# writes two files in the models directory: name.model, and name.vocab
prefix = os.path.join("models", "basic")
tokenizer.save(prefix)

t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")

print(len(tokenizer.merges), "merges")

print("Testing the model")
assert(tokenizer.decode(tokenizer.encode(text)) == text)
print("Success")

print("Testing save/load")
tok = BasicTokenizer()
tok.load(prefix + ".model")
assert(tok.decode(tok.encode(text)) == text)
print("Success")
