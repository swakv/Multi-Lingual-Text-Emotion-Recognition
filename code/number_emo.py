# ['joy' 'fear' 'anger' 'sadness' 'disgust' 'shame' 'guilt']

import pandas as pd
import numpy as np

df = pd.read_csv("../dataset/train_before.csv")
df1 = pd.read_csv("../dataset/val_before.csv")
df2 = pd.read_csv("../dataset/test_before.csv")

# conversion_dict = {
#     'joy' : 0,
#     'fear' : 1,
#     'anger' : 2,
#     'sadness' : 3,
#     'disgust' : 4,
#     'shame' : 5,
#     'guilt' : 6
# }


conversion_dict = {
    'joy' : 0,
    'fear' : 1,
    'anger' : 2,
    'sadness' : 3,
    'disgust' : 4,
    'shame' : 4,
    'guilt' : 4
}

df["Emotion"] = df["Emotion"].apply(lambda x: conversion_dict[x])
df1["Emotion"] = df1["Emotion"].apply(lambda x: conversion_dict[x])
df2["Emotion"] = df2["Emotion"].apply(lambda x: conversion_dict[x])

df = df[df.Emotion != 4]
df1 = df1[df1.Emotion != 4]
df2 = df2[df2.Emotion != 4]

df.to_csv('../dataset/train_modified.csv', index=False)
df1.to_csv('../dataset/val_modified.csv',  index=False)
df2.to_csv('../dataset/test_modified.csv',  index=False)