import streamlit as st
import joblib


# Load trained model
model = joblib.load("movie_genre_model.pkl")


# Page configuration
st.set_page_config(
    page_title="Movie Genre Predictor",
    page_icon="🎬",
    layout="centered"
)


# Title
st.title("🎬 Movie Genre Predictor")

st.write(
    "Enter a movie plot or description below and the machine learning "
    "model will predict its genre."
)


# Input
description = st.text_area(
    "Movie Plot / Description",
    placeholder="Example: A group of friends discover a mysterious haunted house..."
)


# Prediction button
if st.button("Predict Genre"):

    if description.strip() == "":
        st.warning("Please enter a movie description.")

    else:
        prediction = model.predict([description])[0]

        st.success(f"Predicted Genre: **{prediction.upper()}**")


# Project information
st.divider()

st.subheader("📊 About the Model")

st.write(
    "This project uses TF-IDF text vectorization and Logistic Regression "
    "to predict movie genres from plot descriptions."
)

st.write("Model Accuracy: **58.24%**")

st.caption("CODSOFT Machine Learning Internship – Task 1")