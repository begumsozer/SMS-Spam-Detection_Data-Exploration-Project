# sms_spam_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
file_path = "SPAM-text-message.xlsx"
df = pd.read_excel(file_path)

# Basic statistics
print("Class Distribution:\n", df['Category'].value_counts(normalize=True) * 100)

# Visualize distribution
plt.figure(figsize=(6,4))
df['Category'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Distribution of Spam and Ham Messages")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Text preprocessing and feature extraction
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Message'])
y = df['Category']

# Encode labels
y = y.map({'ham': 0, 'spam': 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
