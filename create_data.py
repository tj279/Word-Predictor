import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def create_data():
    # Check for file existence and try alternative paths
    file_path = "HappyDB/happydb/data/cleaned_hm.csv"
    if not os.path.exists(file_path):
        # Try alternative paths
        alternative_paths = [
            "happydb/data/cleaned_hm.csv",
            "HappyDB/data/cleaned_hm.csv",
            "../HappyDB/happydb/data/cleaned_hm.csv"
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                file_path = alt_path
                print(f"Found file at: {alt_path}")
                break
        else:
            print(f"Current directory: {os.getcwd()}")
            print("Available files in HappyDB directory:")
            try:
                print(os.listdir("HappyDB"))
            except:
                print("Could not access HappyDB directory")
            raise FileNotFoundError("Cannot find cleaned_hm.csv file")
    
    # Fix the syntax error in the original code (unbalanced parentheses)
    df = pd.read_csv(file_path)
    df = df['cleaned_hm'].iloc[:1000]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.to_list())
    word_index = tokenizer.word_index

    # Create input data
    inputs = []
    for i in range(0, len(df)):
        sequence = tokenizer.texts_to_sequences([df[i]])[0]
        for j in range(1, len(sequence)):
            inputs.append(sequence[:j+1])

    # Fixed variable name (lenth → length)
    length = 0
    for s in inputs:
        length = max(len(s), length)
    
    inputs = pad_sequences(inputs, maxlen=length, padding='pre')
    inputs = inputs[:, :-1]
    outputs = inputs[:, -1:]
    outputs = to_categorical(outputs, num_classes=len(word_index)+1)
    return inputs, outputs, length, len(word_index)+1,tokenizer
