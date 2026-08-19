import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load training dataset
train_data = pd.read_csv(
    "dataset/train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
)

# Remove missing values
train_data = train_data.dropna(subset=["DESCRIPTION", "GENRE"])

# Input and output
X = train_data["DESCRIPTION"]
y = train_data["GENRE"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model
joblib.dump(model, "movie_genre_model.pkl")

print("\nModel saved successfully as movie_genre_model.pkl")