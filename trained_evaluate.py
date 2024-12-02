import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu import corpus_bleu, corpus_chrf
from rouge_score import rouge_scorer
from bert_score import score
import pandas as pd
import matplotlib.pyplot as plt

# Load trained models and tokenizers
@st.cache_resource
def load_trained_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

# Load trained models
opus_tokenizer, opus_model = load_trained_model_and_tokenizer("./trained_opus_model")
t5_tokenizer, t5_model = load_trained_model_and_tokenizer("./trained_t5_model")

# Function to translate
def translate_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize session state
if "translations" not in st.session_state:
    st.session_state["translations"] = {}
if "evaluations" not in st.session_state:
    st.session_state["evaluations"] = None

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Evaluation", "Results"])

if page == "Evaluation":
    # Evaluation Page
    st.title("Translation Model Evaluation")
    st.write("Compare English-to-German translations using trained models and evaluation metrics.")

    # Input English text
    input_text = st.text_area("Enter English text:", "")

    # Translation button
    if st.button("Translate"):
        if input_text.strip():
            # Translate using both models
            opus_translation = translate_text(opus_model, opus_tokenizer, input_text)
            t5_translation = translate_text(t5_model, t5_tokenizer, input_text)

            # Store translations in session state
            st.session_state["translations"] = {
                "input_text": input_text,
                "opus_translation": opus_translation,
                "t5_translation": t5_translation,
            }

            st.success("Translation Complete!")
        else:
            st.error("Please enter some text to translate!")

    # Display translations if available
    if st.session_state["translations"]:
        translations = st.session_state["translations"]
        st.write("### Translations")
        st.write(f"**Helsinki-NLP Model Translation:** {translations['opus_translation']}")
        st.write(f"**Google-T5 Model Translation:** {translations['t5_translation']}")

        # Reference input
        reference_text = st.text_area("Enter Reference German Translation:", "")

        # Evaluate button
        if st.button("Evaluate Metrics"):
            if reference_text.strip():
                # Evaluate metrics
                opus_translation = translations["opus_translation"]
                t5_translation = translations["t5_translation"]

                # BLEU
                opus_bleu = corpus_bleu([opus_translation], [[reference_text]]).score
                t5_bleu = corpus_bleu([t5_translation], [[reference_text]]).score

                # ROUGE
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                opus_rouge = scorer.score(reference_text, opus_translation)
                t5_rouge = scorer.score(reference_text, t5_translation)

                # ChrF
                opus_chrf = corpus_chrf([opus_translation], [[reference_text]]).score
                t5_chrf = corpus_chrf([t5_translation], [[reference_text]]).score

                # BERTScore
                opus_bert = score([opus_translation], [reference_text], lang="de", verbose=False)
                t5_bert = score([t5_translation], [reference_text], lang="de", verbose=False)

                # Store evaluations in session state
                st.session_state["evaluations"] = {
                    "Metric": ["BLEU", "ROUGE-1 F1", "ROUGE-L F1", "ChrF", "BERTScore F1"],
                    "Helsinki-NLP": [
                        opus_bleu, opus_rouge["rouge1"].fmeasure, opus_rouge["rougeL"].fmeasure, opus_chrf, opus_bert[2].mean().item()
                    ],
                    "Google-T5": [
                        t5_bleu, t5_rouge["rouge1"].fmeasure, t5_rouge["rougeL"].fmeasure, t5_chrf, t5_bert[2].mean().item()
                    ],
                }

                st.success("Evaluation completed. Click on Results for scores.")
            else:
                st.error("Please provide a reference German translation.")

if page == "Results":
    # Results Page
    st.title("Evaluation Results")

    if st.session_state["evaluations"]:
        # Create a DataFrame for evaluation metrics
        eval_data = pd.DataFrame(st.session_state["evaluations"])

        # Display table
        st.write("### Evaluation Results Table")
        st.dataframe(eval_data)

        # Bar graph
        st.write("### Metric Comparison")
        eval_data_melted = eval_data.melt(id_vars=["Metric"], var_name="Model", value_name="Score")

        # Plot
        plt.figure(figsize=(10, 6))
        for index, row in eval_data.iterrows():
            plt.bar(
                row["Metric"],
                row["Helsinki-NLP"],
                label="Helsinki-NLP",
                alpha=0.7,
                width=0.4,
                align='edge'
            )
            plt.bar(
                row["Metric"],
                row["Google-T5"],
                label="Google-T5",
                alpha=0.7,
                width=-0.4,
                align='edge'
            )
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.title("Comparison of Metrics Across Models")
        plt.legend(["Helsinki-NLP", "Google-T5"])
        st.pyplot(plt)
    else:
        st.write("No evaluation data available. Please perform evaluations first.")
