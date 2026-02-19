# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("resumes_30_dataset.csv")

# Label Encoding
label_encoder = LabelEncoder()
df["encoded_role"] = label_encoder.fit_transform(df["target_role"])

# Load BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

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

# Generate embeddings
X = df["resume_text"]
embeddings = torch.vstack([get_embedding(text) for text in X])
y = torch.tensor(df["encoded_role"].values)

# LSTM Classifier
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

model = LSTMClassifier(768, 64, len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    outputs = model(embeddings)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save everything
torch.save(model.state_dict(), "saved_model/lstm_model.pt")

with open("saved_model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

tokenizer.save_pretrained("saved_model/tokenizer")

print("Model Saved Successfully")
