import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# NLTK setup
# ---------------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ---------------------------
# Load dataset (Excel)
# ---------------------------
df = pd.read_excel("attachment_style.xlsx")

# Remove junk columns like Unnamed: 1, 2, ...
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("Available columns:", df.columns)

# ---------------------------
# Column names
# ---------------------------
text_column = "message"
label_column = "style"

# Drop rows with missing labels or text
df = df.dropna(subset=[text_column, label_column])

# ---------------------------
# Text cleaning
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(word for word in text.split() if word not in stop_words)

df["cleaned_text"] = df[text_column].apply(clean_text)

# ---------------------------
# Label encoding
# ---------------------------
df[label_column] = df[label_column].str.strip().str.lower()
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df[label_column])

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_text"],
    df["label_encoded"],
    test_size=0.2,
    random_state=42,
    stratify=df["label_encoded"]
)

# ---------------------------
# TF-IDF Vectorization
# ---------------------------
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# SMOTE (training only)
# ---------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

# ---------------------------
# Logistic Regression + GridSearch
# ---------------------------
param_grid = {
    "C": [0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    LogisticRegression(
        solver="saga",
        max_iter=5000,
        n_jobs=-1
    ),
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro"
)

grid_search.fit(X_train_res, y_train_res)

print("\nBest parameters:", grid_search.best_params_)

# ---------------------------
# Evaluation
# ---------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_vec)

present_labels = sorted(set(y_test))
present_names = le.inverse_transform(present_labels)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=present_names))

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=present_names,
    yticklabels=present_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
