from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from create_data import create_data
from tensorflow.keras.optimizers import Adam
import pickle
import json
import tensorflow as tf

def create_fast_optimized_model(vocab_size, input_length):
    """Create a fast model with targeted optimizations to reduce repetition"""
    model = Sequential()
    
    # Slightly improved embedding
    model.add(Embedding(vocab_size, 200, input_length=input_length))
    
    # Light dropout that won't slow things down much
    model.add(Dropout(0.1))
    
    # First LSTM - same size as original
    # Use the faster CuDNN implementation where possible
    if tf.test.is_gpu_available(cuda_only=True):
        lstm_layer = LSTM(256, return_sequences=True, 
                          dropout=0.1, implementation=2)
    else:
        lstm_layer = LSTM(256, return_sequences=True, dropout=0.1)
    model.add(lstm_layer)
    
    # Second LSTM - same size as original
    if tf.test.is_gpu_available(cuda_only=True):
        lstm_layer2 = LSTM(256, dropout=0.1, implementation=2)
    else:
        lstm_layer2 = LSTM(256, dropout=0.1)
    model.add(lstm_layer2)
    
    # Output layer 
    model.add(Dense(vocab_size, activation='softmax'))
    
    return model

# Get data
inputs, outputs, input_length, vocab_size, tokenizer = create_data()

# Create fast model
model = create_fast_optimized_model(vocab_size, input_length)

# Use Adam with default learning rate
optimizer = Adam(learning_rate=0.001)

# Compile with mixed precision for faster training on GPU
try:
    # Use mixed precision if on TF 2.x
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Using mixed precision")
except:
    print("Mixed precision not available")

# Compile model
model.compile(
    loss='categorical_crossentropy', 
    optimizer=optimizer, 
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Only use early stopping to save time
callbacks = [
    EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

# Train with original parameters
model.fit(
    inputs, outputs,
    epochs=50,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save('word_predictor_model.h5')

# Save tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save metadata
metadata = {
    'vocab_size': vocab_size,
    'input_length': input_length
}

with open('metadata.json', 'w') as f:
    json.dump(metadata, f)

print("Training complete and model saved!")