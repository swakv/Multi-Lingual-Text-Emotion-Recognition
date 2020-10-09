import pandas as pd
import numpy as np

df = pd.read_csv('dataset/ISEAR.csv')
# df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7
train = df[msk]
test = df[~msk]

train.to_csv('dataset/train_before.csv')
test.to_csv('dataset/test_before.csv')