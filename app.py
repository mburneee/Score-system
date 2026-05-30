
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

data = joblib.load("loan_scoring_model.pkl")
model        = data["model"]
scaler       = data["scaler"]
le_emp       = data["label_encoder_employment"]
le_dec       = data["label_encoder_decision"]

EMP_TYPES = le_emp.classes_.tolist()

DECISION_MAP = {
    "approved":      ("approved",  "Зөвшөөрөх",  "Таны зээлийн хүсэлт баталгаажлаа."),
    "manual_review": ("review",    "Гар шалгалт", "Таны хүсэлт нэмэлт шалгалтанд орлоо."),
    "rejected":      ("rejected",  "Татгалзах",   "Таны зээлийн хүсэлт татгалзагдлаа."),
}


def build_features(salary, emp_type, years, amount):
    emp_enc = le_emp.transform([emp_type])[0]
    ratio   = amount / salary
    dti     = amount / (salary * 12)
    log_inc = np.log1p(salary)
    log_amt = np.log1p(amount)
    return np.array([[salary, years, amount, ratio, dti, log_inc, log_amt, emp_enc]])


def predict_score(salary, emp_type, years, amount):
    features = build_features(salary, emp_type, years, amount)
    scaled   = scaler.transform(features)
    proba    = model.predict_proba(scaled)[0]
    pred_idx = model.predict(scaled)[0]

    decision_key = le_dec.classes_[pred_idx]
    # Score: weighted average of class probabilities mapped to 0–1000
    score = round(proba[0] * 1000 + proba[1] * 550 + proba[2] * 150)
    score = int(np.clip(score, 0, 1000))
    return score, decision_key


@app.route("/")
def index():
    return render_template("index.html", emp_types=EMP_TYPES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        salary   = float(request.form["salary"])
        emp_type = request.form["employment"]
        years    = float(request.form["years"])
        amount   = float(request.form["amount"])

        if salary <= 0 or amount <= 0 or years < 0:
            return jsonify({"error": "Утгууд эерэг байх ёстой"}), 400
        if emp_type not in EMP_TYPES:
            return jsonify({"error": "Ажил эрхлэлтийн төрөл буруу байна"}), 400

        score, decision_key = predict_score(salary, emp_type, years, amount)
        status, label, message = DECISION_MAP[decision_key]

        return jsonify({"score": score, "status": status, "label": label, "message": message})
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Оруулсан өгөгдөл буруу: {e}"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
