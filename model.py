from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from create_data import create_data
from tensorflow.keras.optimizers import Adam
import pickle
import json

def create_model(vocab_size, input_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 200, input_length=input_length))
    model.add(LSTM(256))
    model.add(Dense(vocab_size, activation='softmax'))
    return model

# Check if TensorFlow is installed, install if missing
try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not found, installing...")
  
    import tensorflow as tf


inputs,outputs,input_length,vocab_size,tokenizer=create_data()
model = create_model(vocab_size, input_length)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(inputs, outputs, epochs=11, batch_size=256,verbose=1)

model.save('word_predictor_model.h5')

tokenizer = tokenizer  # Get your tokenizer object from your create_data function
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

metadata = {
    'vocab_size': vocab_size,
    'input_length': input_length
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f)