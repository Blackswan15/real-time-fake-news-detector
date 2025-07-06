# 📰 Real-Time Fake News Detection System

A machine learning-based web application to classify news headlines as **FAKE** or **REAL** — in real-time!

---

## ✅ Features

- 🔍 **Trained using 3 real-world datasets** (`fake_or_real_news.csv`, `True.csv`, `Fake.csv`)
- 🧠 **TF-IDF Vectorization** + **LinearSVC Classifier** used for model training
- 🌐 **Live headline validation** using NewsAPI integration
- ✍️ Accepts **user-entered news headlines** and returns a natural verdict (`REAL` or `FAKE`)
- 📈 Achieves **~91% model accuracy** on test data
- ⚡ Built and tested entirely in **JupyterLab**
- 🌍 Developed a FastAPI-based **web interface** for live user input

---

## 🗂 Dataset Sources

- [Kaggle: Fake or Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Combined all 3 CSV files: `fake_or_real_news.csv`, `True.csv`, and `Fake.csv`

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Blackswan15/real-time-fake-news-detector.git

cd real-time-fake-news-detector
pip install -r requirements.txt

python main.py

uvicorn main:app --reload

