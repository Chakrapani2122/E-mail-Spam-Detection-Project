import streamlit as st
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Function to preprocess the input text
def preprocess_text(text):
    # Remove special characters, emojis, and numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    return text

# Load and preprocess the dataset
@st.cache
def load_and_preprocess_data():
    df = pd.read_csv("spam.csv", encoding="ISO-8859-1")
    df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[["label", "message"]]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["message"] = df["message"].apply(preprocess_text)
    return df

# Build and save the pipeline model
@st.cache(allow_output_mutation=True)
def build_and_save_model():
    df = load_and_preprocess_data()
    X = df["message"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pipeline = Pipeline([
        ("tfidfvectorizer", TfidfVectorizer(sublinear_tf=True, encoding="utf-8", decode_error="ignore")),
        ("logisticregression", LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)

    # Save the entire pipeline
    with open("email_spam_detection.pkl", "wb") as model_file:
        pickle.dump(pipeline, model_file)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    return pipeline

# Load the trained pipeline model
try:
    with open("email_spam_detection.pkl", "rb") as model_file:
        pipeline = pickle.load(model_file)
except FileNotFoundError:
    st.warning("Model not found. Building and saving the model...")
    pipeline = build_and_save_model()

# Streamlit app
st.title("Email Spam Detection with Explanation")
st.write("Enter the email content below to check if it's spam or not. The app will also explain why it was classified into that category.")

# Input text area
email_content = st.text_area("Email Content", height=200)

if st.button("Check"):
    if email_content.strip():
        # Preprocess the input text
        email_content_cleaned = preprocess_text(email_content)
        # Transform and predict using the pipeline
        prediction = pipeline.predict([email_content_cleaned])[0]
        prediction_proba = pipeline.predict_proba([email_content_cleaned])[0]

        # Display result
        if prediction == 1:
            st.error("This email is classified as SPAM.")
        else:
            st.success("This email is classified as HAM (Not Spam).")

        # Explanation
        st.subheader("Explanation")
        vectorizer = pipeline.named_steps["tfidfvectorizer"]
        feature_names = vectorizer.get_feature_names_out()
        email_transformed = vectorizer.transform([email_content_cleaned])
        email_features = pd.DataFrame(
            email_transformed.toarray(),
            columns=feature_names
        )
        important_features = email_features.loc[0].sort_values(ascending=False).head(10)
        st.write("Top contributing words for this classification:")
        st.table(important_features)
    else:
        st.warning("Please enter some email content to check.")
