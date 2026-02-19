# model.py

import torch
import torch.nn as nn
import pickle
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load tokenizer & BERT
tokenizer = AutoTokenizer.from_pretrained("saved_model/tokenizer")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

# LSTM Class
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# Load label encoder
with open("saved_model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Initialize model
model = LSTMClassifier(768, 64, len(label_encoder.classes_))
model.load_state_dict(torch.load("saved_model/lstm_model.pt", map_location=torch.device("cpu")))
model.eval()

def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=50
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    return outputs.last_hidden_state[:, 0, :]

def semantic_similarity(resume_text, jd_text):
    resume_vec = get_embedding(resume_text)
    jd_vec = get_embedding(jd_text)

    return cosine_similarity(
        resume_vec.numpy(),
        jd_vec.numpy()
    )[0][0]

def predict_role(resume_text):
    embedding = get_embedding(resume_text)

    with torch.no_grad():
        output = model(embedding)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_role = label_encoder.inverse_transform([predicted_class.item()])[0]

    return predicted_role, confidence.item()

# def ats_screening(resume_text, required_role, jd_text):
#     predicted_role, dl_score = predict_role(resume_text)
#     sim_score = semantic_similarity(resume_text, jd_text)

#     final_score = (dl_score + sim_score) / 2

#     return {
#         "predicted_role": predicted_role,
#         "classification_score": round(dl_score, 3),
#         "semantic_similarity": round(sim_score, 3),
#         "final_score": round(final_score, 3)
#     }
def ats_screening(resume_text, required_role, jd_text):
    predicted_role, dl_score = predict_role(resume_text)
    sim_score = semantic_similarity(resume_text, jd_text)

    final_score = (dl_score + sim_score) / 2

    return {
        "predicted_role": str(predicted_role),
        "classification_score": float(round(dl_score, 3)),
        "semantic_similarity": float(round(sim_score, 3)),
        "final_score": float(round(final_score, 3))
    }
