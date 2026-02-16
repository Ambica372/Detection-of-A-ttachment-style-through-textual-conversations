import gradio as gr
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

# Load saved files
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(word for word in text.split() if word not in stop_words)

def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)
    label = label_encoder.inverse_transform(prediction)
    return label[0]

interface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="Attachment Style Detector"
)

interface.launch()

