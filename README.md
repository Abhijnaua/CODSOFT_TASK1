# 🎬 Movie Genre Classification

A Machine Learning project that predicts the genre of a movie based on its plot description.

## 📌 Project Overview

This project was developed as part of the **CODSOFT Machine Learning Internship – Task 1**.

The model analyzes a movie's plot/description and predicts its genre using Natural Language Processing (NLP) and Machine Learning techniques.

## 🚀 Features

- Predicts movie genre from a plot description
- Uses TF-IDF for text feature extraction
- Uses Logistic Regression for classification
- Interactive Streamlit web application
- Displays the predicted genre instantly

## 🧠 Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- Streamlit

## 📊 Model Performance

**Test Accuracy: 58.24%**

The dataset contains approximately **54,214 movie descriptions** across multiple genres.

## ⚙️ How It Works

1. Movie plot description is entered by the user.
2. TF-IDF converts the text into numerical features.
3. Logistic Regression analyzes the features.
4. The trained model predicts the movie genre.
5. The predicted genre is displayed through the Streamlit application.

## 📂 Project Structure

```text
CODSOFT_TASK1/
│
├── dataset/
│   ├── description.txt
│   ├── test_data.txt
│   └── train_data.txt
│
├── app.py
├── movie_genre.py
├── train_model.py
├── movie_genre_model.pkl
├── requirements.txt
├── .gitignore
└── README.md