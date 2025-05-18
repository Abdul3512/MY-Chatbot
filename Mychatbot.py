
import os
import json
import random
import ssl
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL and NLTK setup
ssl._create_default_https_context = ssl._create_unverified_context
nltk_data_path = os.path.abspath("nltk_data")
nltk.data.path.append(nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)

# Load intents from a JSON file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Preprocess data
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = tags

clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X, y)

# Chatbot function
def chatbot(input_text):
    vec = vectorizer.transform([input_text])
    tag = clf.predict(vec)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

# Streamlit interface
def main():
    st.title("💬 Simple Chatbot")
    st.write("Welcome! Ask me anything related to sports!")

    user_input = st.text_input("You:")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None)

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thanks for chatting!")
            st.stop()

if __name__ == '__main__':
    main()
    
