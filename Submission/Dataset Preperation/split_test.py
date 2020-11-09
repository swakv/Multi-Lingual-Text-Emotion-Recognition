import pandas as pd
import numpy as np

df = pd.read_csv('/Users/Swa/Documents/NTU/Y3S1/CZ4042/Project/code/dataset/test_translated.csv')
# df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.5
train = df[msk]
test = df[~msk]

train.to_csv('/Users/Swa/Documents/NTU/Y3S1/CZ4042/Project/code/dataset/test_before.csv')
test.to_csv('/Users/Swa/Documents/NTU/Y3S1/CZ4042/Project/code/dataset/val_before.csv')