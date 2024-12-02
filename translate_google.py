import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the T5 model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
    return tokenizer, model

tokenizer, model = load_model()

# Function for translation
def translate_text(input_text):
    prompt = f"translate English to German: {input_text}"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Streamlit UI
st.title("English to German Translation App")
st.write("Translate text from English to German using T5!")

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
