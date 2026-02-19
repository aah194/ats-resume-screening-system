# 🧠 ATS Resume Screening System

An AI-powered Applicant Tracking System that classifies resumes into job roles and calculates semantic similarity with job descriptions using BERT + LSTM.

This project demonstrates deep learning model integration into a production-ready backend API.

---

## 🚀 Features

- Resume role classification using BERT embeddings + LSTM
- Semantic similarity scoring between resume and job description
- Combined ATS final scoring
- REST API using Flask
- JSON response ready for frontend integration
- GPU/CPU auto-detection support

---

## 🏗 Tech Stack

- Python
- PyTorch
- HuggingFace Transformers (BERT)
- Scikit-learn
- Flask
- HTML (basic frontend)

---

## 🧠 Model Architecture

1. Resume text → BERT embedding (768-dim vector)
2. Embedding → LSTM classifier
3. Softmax → Predicted role + confidence
4. Cosine similarity → Resume vs Job Description
5. Final score = Average(Classification score + Similarity score)

---

## 📦 How to Run Locally

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

API runs at:
http://127.0.0.1:10000/predict

Sample Request:
{
  "resume": "Experienced Python developer with ML skills",
  "job_description": "Looking for Data Scientist with Python",
  "required_role": "Data Scientist"
}

Sample Response:
{
  "predicted_role": "Machine Learning Engineer",
  "classification_score": 0.285,
  "semantic_similarity": 0.860,
  "final_score": 0.573
}
