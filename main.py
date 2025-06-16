# SMS Spam Detection Project - Decision Systems
# Author: [Your Name]
# Date: [Insert Date]
# Description: Classifies SMS messages as Spam or Ham using a Naive Bayes classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def load_data(path: str):
    #Load and return the SMS spam dataset
    df = pd.read_csv(path, sep='\t', names=['label', 'message'])
    return df

def preprocess(df: pd.DataFrame):
    #Convert labels and split data
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

def vectorize(X_train, X_test):
    #Transform messages into TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer, vectorizer.fit_transform(X_train), vectorizer.transform(X_test)

def train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test):
    #Train the model and print performance
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    dataset_path = "SMSSpamCollection.txt"  # Update with the correct path
    df = load_data(dataset_path)
    
    X_train, X_test, y_train, y_test = preprocess(df)
    vectorizer, X_train_tfidf, X_test_tfidf = vectorize(X_train, X_test)
    
    train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test)
