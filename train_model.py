import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only first two columns
df = df[['v1','v2']]
df.columns = ['label','message']

# Convert labels
df['label'] = df['label'].map({
    'ham':0,
    'spam':1
})

# Features and Target
X = df['message']
y = df['label']

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(X)

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression
model = LogisticRegression()

model.fit(X_train,y_train)

# Accuracy
pred = model.predict(X_test)

print("Accuracy :",accuracy_score(y_test,pred))

# Save model
joblib.dump(model,"spam_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("Model Saved Successfully")