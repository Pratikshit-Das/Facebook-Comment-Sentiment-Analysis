import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Step 1: Load CSV and train model
try:
    df = pd.read_csv('comments.csv', quotechar='"')
    X = df['comment']
    y = df['sentiment']

    # Train TF-IDF and model
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=200)
    model.fit(X_vec, y)

except Exception as e:
    print("❌ Error loading or training model:", e)
    messagebox.showerror("Error", f"Could not load training data: {e}")
    exit()

# Step 2: Function to analyze sentiment
def analyze_sentiment():
    comment = entry.get()
    if not comment.strip():
        messagebox.showwarning("Input Error", "Please enter a comment.")
        return

    comment_vec = vectorizer.transform([comment])
    prediction = model.predict(comment_vec)[0]

    result_label.config(text=f"Sentiment: {prediction.capitalize()}", fg="blue")

# Step 3: Build the GUI
window = tk.Tk()
window.title("Facebook Comment Sentiment Analyzer")
window.geometry("450x200")
window.resizable(False, False)

# Label
label = tk.Label(window, text="Enter a Facebook comment:", font=("Arial", 12))
label.pack(pady=10)

# Entry box
entry = tk.Entry(window, width=50, font=("Arial", 11))
entry.pack(pady=5)

# Analyze button
button = tk.Button(window, text="Analyze Sentiment", command=analyze_sentiment)
button.pack(pady=10)

# Result label
result_label = tk.Label(window, text="", font=("Arial", 14))
result_label.pack(pady=5)

# Start the app
window.mainloop()
