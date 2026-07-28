# 🚀 Customer Satisfaction (CSAT) Prediction using Deep Learning

An end-to-end Deep Learning solution that predicts **Customer Satisfaction (CSAT) scores (1–5)** immediately after a customer support interaction. This project combines advanced feature engineering, Artificial Neural Networks (ANN), embedding layers for high-cardinality categorical features, SHAP explainability, and an interactive Streamlit web application to help businesses identify dissatisfied customers before survey responses are received.

---

## 🎬 Project Demo & Video Explanation

A complete walkthrough of the project covering:

- Business Problem
- Dataset Understanding
- Data Cleaning
- Feature Engineering
- ANN Architecture
- Model Training
- Model Evaluation
- SHAP Explainability
- Business Insights
- Live Streamlit Demonstration

📺 **Watch the complete project explanation here:**

👉 https://drive.google.com/file/d/1qBeM_VqdOKn9gT23V_H7itlTNGB5ghX9/view?usp=sharing

---

## 📌 Project Overview

Customer satisfaction is one of the most important indicators of business success in the e-commerce industry. Traditionally, organizations depend on customer surveys to measure satisfaction. However, many customers never respond to these surveys, making it difficult to identify unhappy customers before they leave the platform.

This project predicts customer satisfaction immediately after a support interaction using historical customer support data. The model enables businesses to proactively identify dissatisfied customers, improve customer retention, and optimize customer support operations.

The project covers the complete machine learning lifecycle from data preprocessing and feature engineering to model deployment and explainable AI.

---

## ✨ Features

- End-to-End Deep Learning Pipeline
- Extensive Data Cleaning & Feature Engineering
- Artificial Neural Network (TensorFlow/Keras)
- Embedding Layers for High Cardinality Features
- SHAP Explainability
- Streamlit Interactive Dashboard
- Real-Time CSAT Prediction
- Business Insights from Model Interpretability

---

## 📊 Dataset

The dataset contains approximately **86,000 customer support interactions** collected from an e-commerce platform.

### Target Variable

- CSAT Score (1–5)

### Important Features

- Support Channel
- Category
- Sub-category
- Customer Remarks
- Order Information
- Customer City
- Product Category
- Item Price
- Support Agent
- Supervisor
- Manager
- Agent Shift
- Issue Reported Time
- Issue Responded Time

---

## ⚙️ Project Workflow

```
Customer Support Dataset
            │
            ▼
     Data Cleaning
            │
            ▼
  Feature Engineering
            │
            ▼
     Data Preprocessing
            │
            ▼
  Deep Learning (ANN)
            │
            ▼
    Model Evaluation
            │
            ▼
   SHAP Explainability
            │
            ▼
   Streamlit Deployment
```

---

## 🛠️ Data Preprocessing

The preprocessing pipeline includes:

- Missing Value Handling
- Duplicate Removal
- Datetime Processing
- Standard Scaling
- One-Hot Encoding
- Label Encoding
- Embedding Preparation
- Data Leakage Prevention

---

## 🧠 Feature Engineering

Several business-driven features were engineered, including:

- Response Time
- Same Day Response
- Weekend Indicator
- Hour of Day
- Day of Week
- Has Order Information
- Customer Remark Length
- Customer Sentiment Score (VADER)
- Has Customer Remark

These engineered features significantly improve the model's ability to understand customer behavior.

---

## 🤖 Deep Learning Architecture

Unlike traditional machine learning models, this project uses a multi-input Artificial Neural Network built with TensorFlow Functional API.

The architecture consists of:

- Dense numerical feature branch
- Agent Name embedding branch
- Customer City embedding branch
- Feature concatenation
- Multiple Dense Layers
- Dropout Regularization
- Softmax Output Layer

Embedding layers efficiently represent high-cardinality categorical variables while reducing dimensionality.

---

## 📈 Model Evaluation

Multiple models and training strategies were evaluated, including:

- Baseline ANN
- Balanced Class Weights
- Manual Class Weights
- Focal Loss
- Oversampling Techniques

The final deployed model was selected based on balanced performance across all CSAT classes rather than overall accuracy.

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- Macro F1 Score
- Quadratic Weighted Kappa
- Confusion Matrix

---

## 🔍 Explainable AI (SHAP)

To improve model transparency, SHAP (SHapley Additive Explanations) was integrated.

The application provides:

- Global Feature Importance
- Local Prediction Explanations
- Top Contributing Features
- Feature Impact Direction

This helps users understand why the model predicted a particular customer satisfaction score.

---

## 💻 Streamlit Application

The deployed application allows users to:

- Enter customer interaction details
- Predict CSAT Score in real time
- View prediction probabilities
- Understand prediction using SHAP explanations

---

## 📷 Application Preview

> Add screenshots inside the `images/` folder and update the paths below.

```
images/dashboard.png

images/prediction.png

images/shap_explanation.png
```

---

## 🧰 Tech Stack

### Programming Language

- Python

### Machine Learning & Deep Learning

- TensorFlow
- Keras
- Scikit-learn

### Data Processing

- Pandas
- NumPy

### Explainable AI

- SHAP

### NLP

- VADER Sentiment Analysis

### Deployment

- Streamlit

### Visualization

- Matplotlib
- Plotly

---

## 📁 Project Structure

```
customer-satisfaction-prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│
├── notebooks/
│
├── preprocessing_objects/
│   ├── scaler.pkl
│   ├── agent_encoder.pkl
│   ├── city_encoder.pkl
│   ├── preprocessing_metadata.pkl
│   ├── onehot_column_list.pkl
│   ├── shap_background.pkl
│
├── models/
│   └── csat_ann_model.keras
│
└── images/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-satisfaction-prediction.git
```

Move into the project folder

```bash
cd customer-satisfaction-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

- REST API Deployment
- Docker Containerization
- MLflow Experiment Tracking
- Cloud Deployment (AWS/Azure/GCP)
- Model Monitoring
- CI/CD Pipeline
- Real-Time Data Integration

---

## 💼 Business Impact

This project demonstrates how deep learning can help organizations:

- Identify dissatisfied customers early
- Reduce customer churn
- Improve customer support quality
- Understand key drivers of customer satisfaction
- Enable proactive customer engagement

---

## 📚 Key Learnings

- Building production-ready deep learning pipelines
- Handling high-cardinality categorical variables using embeddings
- Feature engineering for customer analytics
- Explainable AI using SHAP
- End-to-end deployment with Streamlit
- Business-oriented machine learning development

---

## 👨‍💻 Author

**Suryanshu Gupta**


