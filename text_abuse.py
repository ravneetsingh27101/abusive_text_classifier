import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox
import re

# Global variables (initially empty)
vectorizer = None
lr_model = None
svm_model = None
rfc_model = None
lr_accuracy = None
svm_accuracy = None
rfc_accuracy = None

# Expanded abusive word list
abusive_words = [
    "idiot",
    "stupid",
    "fool",
    "racist",
    "hate",
    "moron",
    "dumb",
    "ass",
    "loser",
    "jerk",
    "retard",
    "ugly",
    "fat",
    "bitch",
    "shit",
]


# Function to train the models
def train_models():
    global vectorizer, lr_model, svm_model, rfc_model
    global lr_accuracy, svm_accuracy, rfc_accuracy

    try:
        # Load the dataset
        data = pd.read_csv(
            "C:\\Users\\ASUS\\Desktop\\project\\twitter_racism_parsed_dataset.csv"
        )

        # Data preprocessing - ensure text is string type
        data["text"] = data["text"].astype(str)

        # Improve vectorization by focusing just on the text
        vectorizer = TfidfVectorizer(
            max_features=2000, ngram_range=(1, 3), min_df=2, stop_words="english"
        )
        X = vectorizer.fit_transform(data["text"]).toarray()
        y = data["label"]

        # Split data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Logistic Regression with enhanced parameters
        lr_model = LogisticRegression(
            solver="liblinear", C=1.0, class_weight="balanced"
        )
        lr_model.fit(X_train, y_train)
        lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))

        # Train SVM with enhanced parameters
        svm_model = SVC(
            kernel="linear", C=1.0, class_weight="balanced", probability=True
        )
        svm_model.fit(X_train, y_train)
        svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))

        # Train Random Forest with enhanced parameters
        rfc_model = RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight="balanced", random_state=42
        )
        rfc_model.fit(X_train, y_train)
        rfc_accuracy = accuracy_score(y_test, rfc_model.predict(X_test))

        messagebox.showinfo(
            "Training Complete",
            f"Models trained successfully!\n\n"
            f"Logistic Regression Accuracy: {lr_accuracy*100:.2f}%\n"
            f"SVM Accuracy: {svm_accuracy*100:.2f}%\n"
            f"Random Forest Accuracy: {rfc_accuracy*100:.2f}%",
        )

        # Enable Predict button after training
        predict_button.config(state="normal")

    except Exception as e:
        messagebox.showerror("Error", f"Training failed: {e}")


# Enhanced function to find abusive words in the sentence
def find_abusive_words(sentence):
    # Convert to lowercase and tokenize by splitting on non-alphanumeric characters
    words = re.findall(r"\b\w+\b", sentence.lower())
    found_abusive = [word for word in words if word in abusive_words]
    return found_abusive


# Function to predict whether a sentence is abusive or not
def predict_abuse(sentence):
    # Check for keyword-based detection first
    abusive_found = find_abusive_words(sentence)
    keyword_result = len(abusive_found) > 0

    # Process with ML models
    input_vector = vectorizer.transform([sentence]).toarray()

    lr_prediction = lr_model.predict(input_vector)[0]
    svm_prediction = svm_model.predict(input_vector)[0]
    rfc_prediction = rfc_model.predict(input_vector)[0]

    # If abusive words found, override model predictions that say non-abusive
    if keyword_result:
        return {
            "Logistic Regression": "Abusive",
            "SVM": "Abusive",
            "Random Forest": "Abusive",
            "abusive_words": abusive_found,
            "keyword_detection": True,
        }
    else:
        return {
            "Logistic Regression": "Abusive" if lr_prediction == 1 else "Non-abusive",
            "SVM": "Abusive" if svm_prediction == 1 else "Non-abusive",
            "Random Forest": "Abusive" if rfc_prediction == 1 else "Non-abusive",
            "abusive_words": [],
            "keyword_detection": False,
        }


# Predict button action
def on_predict():
    sentence = entry.get()
    if not sentence.strip():
        messagebox.showwarning("Input Error", "Please enter a sentence.")
        return

    predictions = predict_abuse(sentence)

    # Get detected abusive words
    abusive_found = predictions.get("abusive_words", [])

    # Get models that flagged as abusive
    abusive_models = [
        model
        for model, result in predictions.items()
        if result == "Abusive"
        and model != "abusive_words"
        and model != "keyword_detection"
    ]

    # Build the result text
    result_content = [
        f"Logistic Regression: {predictions['Logistic Regression']}",
        f"SVM: {predictions['SVM']}",
        f"Random Forest: {predictions['Random Forest']}",
    ]

    if predictions["keyword_detection"]:
        result_content.append(f"\n⚠ ABUSIVE CONTENT DETECTED")
        result_content.append(f"Detected abusive words: {', '.join(abusive_found)}")
        result_content.append("👉 Consider replacing with more respectful language.")
    elif abusive_models:
        result_content.append(
            f"\n⚠ Flagged as potentially abusive by: {', '.join(abusive_models)}"
        )
    else:
        result_content.append("\n✅ All models agree: Non-abusive")

    result_text.set("\n".join(result_content))


# Show model accuracies
def show_accuracy():
    if lr_model is None:
        messagebox.showwarning("Warning", "Please train the models first.")
        return

    messagebox.showinfo(
        "Model Accuracies",
        f"Logistic Regression: {lr_accuracy*100:.2f}%\n"
        f"SVM: {svm_accuracy*100:.2f}%\n"
        f"Random Forest: {rfc_accuracy*100:.2f}%",
    )


# Tkinter GUI setup
window = tk.Tk()
window.title("Abusive Text Classifier")
window.geometry("500x450")

# Title Label
title_label = tk.Label(
    window, text="Abusive Text Classifier", font=("Helvetica", 16, "bold")
)
title_label.pack(pady=10)

# Train Button
train_button = tk.Button(
    window, text="Train Models", command=train_models, width=20, bg="blue", fg="white"
)
train_button.pack(pady=5)

# Input Field
entry = tk.Entry(window, width=50, font=("Helvetica", 12))
entry.pack(pady=10)

# Predict Button (disabled until training)
predict_button = tk.Button(
    window,
    text="Predict",
    command=on_predict,
    width=15,
    bg="green",
    fg="white",
    state="disabled",
)
predict_button.pack(pady=5)

# Result Label
result_text = tk.StringVar()
result_label = tk.Label(
    window,
    textvariable=result_text,
    font=("Helvetica", 12),
    justify="left",
    wraplength=450,
)
result_label.pack(pady=10)

# Accuracy Button
accuracy_button = tk.Button(
    window, text="Show Model Accuracies", command=show_accuracy, width=25
)
accuracy_button.pack(pady=10)

# Run the GUI
window.mainloop()
