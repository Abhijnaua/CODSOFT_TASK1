import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
print("Loading dataset...")

df = pd.read_csv(
    "dataset/train_data.txt",
    sep=":::",
    engine="python",
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
)

# Remove missing values
df = df.dropna(subset=["DESCRIPTION", "GENRE"])

print("Dataset loaded successfully!")
print("Number of records:", len(df))


# Input and output
X = df["DESCRIPTION"]
y = df["GENRE"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create machine learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# Train model
print("\nTraining the model...")
model.fit(X_train, y_train)


# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# Save model
with open("movie_genre_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as movie_genre_model.pkl")