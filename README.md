# 🫀 S.S_AI — Heart Disease Prediction

An AI-powered web app that predicts the risk of heart disease based on patient clinical data, built as part of the **CodeAlpha Machine Learning Internship — Task 4**.

## 🎯 WHAT
A Streamlit web application that takes patient medical parameters (age, chest pain type, cholesterol, blood pressure, etc.) as input and predicts whether the patient is at risk of heart disease, along with a probability score.

## ❓ WHY
Early detection of heart disease risk can help in timely medical intervention. This project demonstrates how machine learning can assist in preliminary health risk screening using structured clinical data.

## ⚙️ HOW
- **Dataset:** UCI Heart Disease Dataset (920 patient records, 4 hospital sources — Cleveland, Hungary, Switzerland, VA Long Beach)
- **Preprocessing:** Missing value imputation (median/mode), Label Encoding for categorical features, feature scaling with StandardScaler
- **Models Trained & Compared:** Logistic Regression, Decision Tree, Random Forest
- **Final Model:** Random Forest Classifier (best performing)
- **Deployment:** Streamlit Cloud

## 📊 RESULT
| Metric | Score |
|---|---|
| Accuracy | 83.2% |
| Precision | 83.2% |
| Recall | 87.3% |
| F1-Score | 85.2% |
| ROC-AUC | 0.92 |

## 🚀 Live Demo
[Add your Streamlit Cloud link here after deployment]

## 🛠️ Tech Stack
Python, Pandas, NumPy, Scikit-learn, Streamlit

## 📥 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ⚕️ Disclaimer
This tool is for educational and portfolio purposes only and is not a substitute for professional medical diagnosis.

---
Built with ❤️ by **S.S_AI**