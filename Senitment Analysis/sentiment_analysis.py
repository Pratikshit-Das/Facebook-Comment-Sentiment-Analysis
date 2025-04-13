import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# Optional: from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Step 1: Load the dataset (handles quoted commas correctly)
df = pd.read_csv('comments.csv', quotechar='"')

# Step 2: Display sentiment class distribution
print("📊 Class distribution:\n")
print(df['sentiment'].value_counts(), "\n")

# Step 3: Define input and output
X = df['comment']
y = df['sentiment']

# Step 4: Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 5: Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train the model
model = LogisticRegression(max_iter=200)
# model = MultinomialNB()  # Alternative model for small datasets
model.fit(X_train_vec, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test_vec)
print("\n🔍 Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=1))

# Step 8: Show actual vs predicted
print("📋 Actual vs Predicted:")
for comment, true, pred in zip(X_test, y_test, y_pred):
    print(f"Comment: {comment} | Actual: {true} | Predicted: {pred}")

# Step 9: Predict sentiment on new comments
new_comments = [
    "I hate this feature!",
    "Really good update!",
    "Service was fine, nothing special.",
    "Absolutely amazing support team!",
    "This is the worst app I’ve ever used.",
    "Super fast and reliable.",
    "Nothing special. Just works.",
    "Terrible design!"
]

new_vec = vectorizer.transform(new_comments)
new_preds = model.predict(new_vec)

print("\n🆕 New Comments Predictions:")
for comment, sentiment in zip(new_comments, new_preds):
    print(f"{comment} => {sentiment}")
