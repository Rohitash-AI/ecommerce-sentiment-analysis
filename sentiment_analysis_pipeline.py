import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

# Ensure NLTK data is downloaded (uncomment if running for the first time)
# nltk.download('stopwords')
# nltk.download('wordnet')

print("1. Loading Data...")
df = pd.read_csv('customer_reviews_raw.csv')

print("\n2. Data Cleaning (Pandas)...")
# Handle missing values
df = df.dropna(subset=['Review'])
df['Product'] = df['Product'].replace('', 'Unknown Product')
# Fix outlier ratings
df['Rating'] = df['Rating'].apply(lambda x: x if 1 <= x <= 5 else np.nan)
df['Rating'] = df['Rating'].fillna(df['Rating'].median())

print(f"Data shape after cleaning: {df.shape}")

print("\n3. NLP Preprocessing...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove special characters and lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    # Tokenize, remove stopwords, and lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['Clean_Review'] = df['Review'].apply(clean_text)

print("\n4. Machine Learning: Sentiment Classification...")
# We will train a Logistic Regression model to predict sentiment from the text

# Prepare X (features) and y (target)
X = df['Clean_Review']
y = df['Sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n5. Data Visualization (Matplotlib & Seaborn)...")
# Visualization 1: Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=['Positive', 'Neutral', 'Negative'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Positive', 'Neutral', 'Negative'],
            yticklabels=['Positive', 'Neutral', 'Negative'])
plt.title('Sentiment Classification Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')

# Visualization 2: Average Rating by Product
plt.figure(figsize=(12, 6))
avg_rating = df.groupby('Product')['Rating'].mean().sort_values(ascending=False)
sns.barplot(x=avg_rating.values, y=avg_rating.index, palette='viridis')
plt.title('Average Customer Rating by Product')
plt.xlabel('Average Rating (1-5)')
plt.ylabel('Product')
plt.tight_layout()
plt.savefig('product_ratings.png')

print("Saved 'confusion_matrix.png' and 'product_ratings.png'.")
