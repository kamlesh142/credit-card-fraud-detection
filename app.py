from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib

# IMPORTANT FIX FOR FLASK + MATPLOTLIB
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# LOAD MODEL
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
accuracy = joblib.load("accuracy.pkl")

# FEATURE NAMES
FEATURES = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4']


@app.route('/')
def home():
    return render_template(
        'index.html',
        accuracy=round(accuracy * 100, 2)
    )


@app.route('/predict', methods=['POST'])
def predict():

    try:

        # GET RAW INPUTS
        raw_values = [
            request.form['Time'],
            request.form['Amount'],
            request.form['V1'],
            request.form['V2'],
            request.form['V3'],
            request.form['V4']
        ]

        # CHECK EMPTY INPUTS
        for value in raw_values:
            if value.strip() == "":
                return render_template(
                    'index.html',
                    prediction_text="⚠ Please fill all input fields.",
                    accuracy=round(accuracy * 100, 2)
                )

        # CONVERT TO FLOAT
        values = [float(v) for v in raw_values]

        # DATAFRAME WITH FEATURE NAMES
        features = pd.DataFrame([values], columns=FEATURES)

        # SCALE DATA
        scaled = scaler.transform(features)

        # PREDICTION
        pred = model.predict(scaled)

        # REAL PROBABILITY
        prob = model.predict_proba(scaled)[0]

        normal = round(prob[0], 2)
        fraud = round(prob[1], 2)

        # RESULT
        if pred[0] == 1:
            result = "🚨 FRAUD TRANSACTION DETECTED"
            explanation = (
                "High risk transaction detected due to unusual "
                "behavior pattern in PCA features."
            )

        else:
            result = "✅ NORMAL TRANSACTION"
            explanation = (
                "Transaction matches normal spending behavior."
            )

        # PIE CHART
        plt.figure(figsize=(4, 4))

        plt.pie(
            [normal, fraud],
            labels=['Normal', 'Fraud'],
            autopct='%1.1f%%',
            colors=['#00b894', '#d63031']
        )

        plt.title("Fraud Detection Result")

        # SAVE TO MEMORY
        buf = BytesIO()

        plt.savefig(buf, format='png', bbox_inches='tight')

        buf.seek(0)

        chart = base64.b64encode(buf.read()).decode('utf-8')

        # CLOSE FIGURE
        plt.close()

        return render_template(
            'index.html',
            prediction_text=result,
            chart=chart,
            explanation=explanation,
            accuracy=round(accuracy * 100, 2),
            normal=round(normal * 100, 2),
            fraud=round(fraud * 100, 2)
        )

    except ValueError:

        return render_template(
            'index.html',
            prediction_text="⚠ Invalid input. Enter numeric values only.",
            accuracy=round(accuracy * 100, 2)
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f"⚠ Error: {str(e)}",
            accuracy=round(accuracy * 100, 2)
        )
if __name__ == "__main__":
    print("Server Running...")
    app.run(host="127.0.0.1", port=5000, debug=True)    