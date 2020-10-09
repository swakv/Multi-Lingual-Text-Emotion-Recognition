import pandas as pd
import numpy as np
from googletrans import Translator
import random

translator = Translator()

def translateSentence(sentence):
#   languages = ['es', 'hi', 'zh-CN', 'ja', 'ta','ko', 'fr', 'ru', 'it' ,'de', 'vi']
    languages = ['hi']
    translation = translator.translate(sentence, dest=random.choice(languages))
    return translation.text


df = pd.read_csv("/Users/Swa/Documents/NTU/Y3S1/CZ4042/Project/code/dataset/test_initial.csv")

df["Comment"] = df["Comment"].apply(translateSentence)

df.to_csv('/Users/Swa/Documents/NTU/Y3S1/CZ4042/Project/code/dataset/test_translated.csv')







