import streamlit as st
from transformers import pipeline

# Load the Large Language Model (LLM)
llm = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

# Title
st.title("Large Language Model Interface")

# User Input
user_input = st.text_input("Enter text:", "Once upon a time,")

# Generate Text
if st.button("Generate Text"):
    generated_text = llm(user_input, max_length=100)[0]['generated_text']
    st.text(generated_text)
