import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

#import dataset
def create_data():
    df=pd.read_csv("D:\Projects\Word Predictor\HappyDB\happydb\data\cleaned_hm.csv"
               
)
    df=df['cleaned_hm'].iloc[:10000]
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(df.to_list())
    word_index=tokenizer.word_index

    #create input data
    inputs=[]
    for i in range(0, len(df)):
        sequence=tokenizer.texts_to_sequences([df[i]])[0]
        for j in range(1, len(sequence)):
            inputs.append(sequence[:j+1])

    lenth=0
    for s in inputs:
        lenth=max(len(s), lenth)
    
    inputs=pad_sequences(inputs, maxlen=lenth, padding='pre')

    inputs=inputs[:, :-1]
    outputs=inputs[:, -1:]
    outputs=to_categorical(outputs, num_classes=len(word_index)+1)
    return inputs, outputs,lenth,len(word_index)+1
