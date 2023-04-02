pip install nlu
import streamlit as st
import nlu
import spacy
from transformers import MarianMTModel, MarianTokenizer

# Disable Streamlit's caching
st.set_page_config( \
    page_title="Ancient Greek Lemmatizer", \
    page_icon=":books:", \
    layout="wide", \
    initial_sidebar_state="expanded" \
)
st.set_option('deprecation.showfileUploaderEncoding', False) # This is to suppress a warning message

# Load the English and Greek models
english_nlp = spacy.load("en_core_web_sm")
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-el")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-el")

def translate_lemmatize_translate(text):
    # Translate the Greek text to English
    english_df = nlu.load('xx.grk.marian.translate_to.en').predict([text], output_level='sentence')
    
    # Check if the 'translated' key is present in the dictionary
    if 'translated' not in english_df:
        st.error("Error: 'translated' key not found in dictionary")
        return None
    
    english_text = english_df['translated'][0]
    
    # Lemmatize the English text
    english_doc = english_nlp(english_text)
    english_lemmas = [token.lemma_ for token in english_doc]
    english_lemmas_str = " ".join(english_lemmas) # Join the lemmas into a single string
    
    # Translate the lemmas to Greek
    tokenized = tokenizer.prepare_seq2seq_batch([english_lemmas_str], return_tensors="pt")
    translated = model.generate(**tokenized)
    ancient_greek_lemmas = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    
    return ancient_greek_lemmas

# Set the app title
st.title("Ancient Greek Lemmatizer")

# Add a text input box for the user to enter the Greek text
text_input = st.text_input("Enter Greek text:")

# Add a button to trigger the translation and lemmatization process
if st.button("Translate and Lemmatize"):
    # Call the translate_lemmatize_translate function with the user's input
    ancient_greek_lemmas = translate_lemmatize_translate(text_input)
    
    # Display the lemmas
    if ancient_greek_lemmas is not None:
        st.success(ancient_greek_lemmas)
