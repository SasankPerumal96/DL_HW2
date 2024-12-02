import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the English-to-German model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    return tokenizer, model

tokenizer, model = load_model()

# Function for translation
def translate_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Streamlit UI
st.title("English to German Translation App")
st.write("Translate text from English to German using Helsinki-NLP's Opus-MT model!")

# Input text box
input_text = st.text_area("Enter English text to translate:", "")

if st.button("Translate"):
    if input_text.strip():
        with st.spinner("Translating..."):
            translation = translate_text(input_text)
        st.success("Translation Complete!")
        st.write("### Translated Text:")
        st.write(translation)
    else:
        st.error("Please enter some text to translate!")
