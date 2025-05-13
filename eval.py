import tensorflow as tf
import numpy as np
import pickle



# Load the saved model
model = tf.keras.models.load_model('word_predictor_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
def generate_text(seed_text, next_words, model, tokenizer, temperature=1.0):
    # Correctly get sequence length from model's input shape
    max_sequence_len = model.input_shape[1] + 1
    output_text = seed_text
    
    for _ in range(next_words):
        # Tokenize the current text correctly
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        
        if len(token_list) > max_sequence_len - 1:
            token_list = token_list[-(max_sequence_len-1):]
            
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list], maxlen=max_sequence_len-1, padding='pre'
        )
        
        # Get prediction probabilities
        predicted_probs = model.predict(padded_sequence, verbose=0)[0]
        
        # Apply temperature with safety for log(0)
        predicted_probs = np.log(predicted_probs + 1e-10) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)
        
        # Ensure sum is exactly 1.0 to avoid floating point precision errors
        predicted_probs = predicted_probs / np.sum(predicted_probs)
        
        # Alternative sampling method that avoids multinomial issues
        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        
        # Convert the predicted index back to a word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
                
        output_text += " " + output_word
        
    return output_text

# Try different temperature values
print("Temperature = 0.7 (More Conservative):")
print(generate_text('I love life', 50, model, tokenizer, temperature=0.7))

print("\nTemperature = 1.0 (Balanced):")
print(generate_text('I love life', 50, model, tokenizer, temperature=1.0))

print("\nTemperature = 1.5 (More Creative):")
print(generate_text('I love life', 50, model, tokenizer, temperature=1.5))
