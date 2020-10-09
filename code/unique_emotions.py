import pandas as pd
import numpy as np

df = pd.read_csv('/Users/Swa/Documents/NTU/Y3S1/CZ4042/Project/code/dataset/test_before.csv')
print(df["Emotion"].unique())
# ['joy' 'fear' 'anger' 'sadness' 'disgust' 'shame' 'guilt']