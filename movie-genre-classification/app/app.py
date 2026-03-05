import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="centered"
)

# ----------------------------
# Load Model & Tokenizer
# ----------------------------

MODEL_PATH = "final_model"

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

model.eval()

# ----------------------------
# Load Dataset to Recreate Label Mapping
# ----------------------------

df = pd.read_csv(
    "train_data.txt",
    sep=" ::: ",
    engine="python",
    header=None
)

df.columns = ["id", "title", "genre", "plot"]

label_encoder = LabelEncoder()
label_encoder.fit(df["genre"])

id2label = {i: label for i, label in enumerate(label_encoder.classes_)}

# ----------------------------
# Prediction Function
# ----------------------------

def predict_genre(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=1).squeeze()

    top_probs, top_indices = torch.topk(probabilities, 5)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append((id2label[idx.item()], prob.item() * 100))

    return results

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🎬 Movie Genre Classification App")
st.write("Enter a movie plot and the model will predict the genre.")

user_input = st.text_area("✍ Enter Movie Plot:", height=200)

if st.button("Predict Genre"):

    if user_input.strip() == "":
        st.warning("Please enter a movie plot.")
    else:
        results = predict_genre(user_input)

        genres = [r[0] for r in results]
        scores = [r[1] for r in results]

        st.success(f"🎯 Predicted Genre: **{genres[0]}**")
        st.info(f"Confidence: {round(scores[0], 2)}%")

        # Bar Chart
        fig, ax = plt.subplots()
        ax.barh(genres[::-1], scores[::-1])
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Top 5 Genre Predictions")

        st.pyplot(fig)