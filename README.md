Save this as README.md in your GitHub project.

# Credit Card Fraud Detection System
A Machine Learning based web application that detects fraudulent credit card transactions using Flask and Scikit-learn.
## 🚀 Project Overview
This project predicts whether a credit card transaction is:
- ✅ Normal Transaction
- 🚨 Fraudulent Transaction
The system uses a trained Machine Learning model and displays prediction results with probability visualization using pie charts.
---
## 🛠 Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- Matplotlib
- Joblib
- HTML/CSS
---
## 📂 Project Structure
```bash
credit-card-fraud-detection/
│
├── app.py
├── fraud_model.pkl
├── scaler.pkl
├── accuracy.pkl
├── requirements.txt
├── Procfile
│
├── templates/
│   └── index.html
│
└── README.md

⸻

⚙ Features

* Fraud detection using ML model
* User-friendly web interface
* Real-time prediction
* Fraud probability visualization
* Error handling for invalid inputs
* Deployable on Render

⸻

📊 Input Features

The model uses the following features:

* Time
* Amount
* V1
* V2
* V3
* V4

⸻

▶ How to Run Locally

1. Clone Repository

git clone https://github.com/yourusername/credit-card-fraud-detection.git

2. Open Project Folder

cd credit-card-fraud-detection

3. Install Requirements

pip install -r requirements.txt

4. Run Flask App

python app.py

5. Open Browser

http://127.0.0.1:5000

⸻

☁ Deployment

This project can be deployed easily on:

* Render
* Railway
* Heroku

Render Start Command

gunicorn app:app

⸻

📈 Model Accuracy

The project displays trained model accuracy dynamically on the homepage.

⸻

🧠 Machine Learning

The fraud detection model was trained using:

* Classification Algorithm
* Feature Scaling
* Probability Prediction

⸻

📸 Output

The application shows:

* Prediction result
* Fraud percentage
* Normal percentage
* Pie chart visualization

⸻

👨‍💻 Author

Kamlesh Yadav

⸻

📄 License

This project is for educational and learning purposes.
