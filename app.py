import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import tensorflow.compat.v1 as tf

# Suppress deprecated warnings
tf.compat.v1.disable_v2_behavior()
# Load the LSTM Model
model_path = './next_word_GRU.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error("Model file not found. Please make sure 'next_word_GRU.h5' is in the correct directory.")
    st.stop()

# Load the tokenizer
tokenizer_path = 'tokenizer.pickle'
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    st.error("Tokenizer file not found. Please make sure 'tokenizer.pickle' is in the correct directory.")
    st.stop()

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app design
st.set_page_config(page_title="Next Word Prediction", page_icon="🔮", layout="centered")

# Main page title and description
st.title("🔮 Next Word Prediction with GRU ")
st.markdown("""
<style>
    .main-title {
        font-family: 'Arial', sans-serif;
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .text-box {
        border-radius: 10px;
        border: 1px solid #1f77b4;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.write("### Enter a sequence of words to predict the next word 🌐")
st.write("This app uses an GRU model to predict the next word in the input sequence.")

# Input box for user text
input_text = st.text_input("Input your sequence of words here:", "To be or not to")

# Button for prediction
if st.button("💡 Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    if next_word:
        st.success(f'Next word: **{next_word}**')
    else:
        st.warning("Sorry, no prediction could be made. Please try a different input.")

# Add some color to the UI
st.markdown("""
<style>
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton button:hover {
        background-color: #0f4c81;
        transition: 0.3s;
    }
    .stTextInput input {
        border-radius: 8px;
        border: 2px solid #1f77b4;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Footer section
st.markdown("""
---
*Developed by [Ekansh Sharma]. This application showcases a simple GRU-based next word prediction model. Enjoy exploring!*
""")
