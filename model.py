from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from create_data import create_data
from tensorflow.keras.optimizers import Adam

def create_model(vocab_size, input_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=input_length))
    model.add(LSTM(150))
    model.add(Dense(vocab_size, activation='softmax'))
    return model

inputs,outputs,input_length,vocab_size=create_data()
model = create_model(vocab_size, input_length)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(inputs, outputs, epochs=10, batch_size=64,verbose=1)