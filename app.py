import streamlit as st
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Load model & vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("📝 Sentiment Analysis App")
st.write("Enter a product review to predict sentiment.")

user_input = st.text_area("Enter Review")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.success("Sentiment: **Positive 😊**")
    else:
        st.error("Sentiment: **Negative 😡**")
