# app.py

from flask import Flask, request, jsonify
from model import ats_screening

app = Flask(__name__)

@app.route("/")
def home():
    return "ATS Resume Screening API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    resume_text = data["resume"]
    job_description = data["job_description"]
    required_role = data["required_role"]

    result = ats_screening(resume_text, required_role, job_description)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
