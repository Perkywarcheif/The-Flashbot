import os
import json
import datetime
import pandas as pd
import random
import joblib
import streamlit as st

# Load intents from JSON file
@st.cache_resource
def load_intents():
    with open("intents.json", "r") as file:
        return json.load(file)

# Load pre-trained model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    clf = joblib.load("model.pkl")  # Pre-trained Logistic Regression model
    vectorizer = joblib.load("vectorizer.pkl")  # Pre-trained TfidfVectorizer
    return clf, vectorizer

# Chatbot response function
def chatbot(input_text, clf, vectorizer, intents):
    input_vector = vectorizer.transform([input_text])
    tag = clf.predict(input_vector)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Main function
def main():
    # Load resources
    intents = load_intents()
    clf, vectorizer = load_model_and_vectorizer()

    # Streamlit UI
    st.title("Fast Chatbot with NLP")
    st.sidebar.title("Menu")
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome! Type a message to chat.")

        # Initialize session state for conversation history
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []

        # User input
        user_input = st.text_input("You:", key="user_input")

        if st.button("Send", key="send_button"):
            if user_input:
                # Get chatbot response
                response = chatbot(user_input, clf, vectorizer, intents)

                # Append conversation to session state
                st.session_state.conversation.append((user_input, response))

                # Display conversation
                for user_msg, bot_msg in st.session_state.conversation:
                    st.write(f"*You:* {user_msg}")
                    st.write(f"*Chatbot:* {bot_msg}")

                # Stop if chatbot says goodbye
                if response.lower() in ['goodbye', 'bye']:
                    st.write("Chatbot: Thank you for chatting!")
                    st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        if 'conversation' in st.session_state and st.session_state.conversation:
            for user_msg, bot_msg in st.session_state.conversation:
                st.write(f"*You:* {user_msg}")
                st.write(f"*Chatbot:* {bot_msg}")
        else:
            st.write("No conversation history available.")

    elif choice == "About":
        st.header("About")
        st.write("This chatbot uses NLP and Logistic Regression to respond based on predefined intents.")

# Run the app
if __name__ == "__main__":
    main()