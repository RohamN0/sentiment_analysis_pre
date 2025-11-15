# ⭐ Real-Time Review Score Predictor (1--5)

This project is a **real-time product review rating predictor** built
with **Streamlit** and powered by **transformer-based models**.\
Given any review text, the app predicts a **star rating from 1 to 5**
along with a **confidence breakdown** for each class.

## 🚀 Features

-   🧠 Predicts **1--5 star ratings** based on review text\
-   📊 Displays **confidence probabilities** for all rating classes\
-   ⚡ Real-time inference using **HuggingFace Transformers**\
-   🧹 Includes a text preprocessing pipeline\
-   🎨 Clean Streamlit UI

## 🧩 Model Information

This project allows you to load ANY transformer model you prefer.

### 🔹 Default Model Used

The included example uses:

**roberta-large**\
with weights loaded from a locally stored `model.bin`.

### 🔹 Alternate Model Source

You can also use the model provided by:

https://github.com/saraM0radi/Sentiment_Analysis

This model achieves **77% accuracy** and is fully compatible with this
project.

## 📁 Project Structure

    📦 sentiment-rating-app
     ┣ 📜 app.py
     ┣ 📜 model.py
     ┣ 📜 model.bin
     ┣ 📜 README.md

## ▶️ How to Run

### 1️⃣ Install dependencies

    pip install transformers torch streamlit pandas

### 2️⃣ Run Streamlit

    streamlit run app.py

## 🛠 How It Works

### **TextPipeline**

Handles: - Lowercasing\
- Expanding contractions\
- Cleaning spaces\
- Tokenization

## 📜 License

MIT License
