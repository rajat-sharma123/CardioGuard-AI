# 🩺 CardioGuard: Heart Disease Risk Assessment System

An end-to-end Machine Learning web application designed for cardiovascular risk stratification and clinical decision support. Built using **Python**, **Scikit-Learn**, and **Streamlit**.

---

## 🌟 Key Highlights & Project Features

1. **Multi-Model Benchmark & Comparison**:
   - **Random Forest Classifier** (Primary ensemble model — 89.1% Accuracy, 89.6% Recall, 95.6% ROC-AUC)
   - **Logistic Regression** (Linear baseline with feature normalization — 88.8% Accuracy, 96.3% ROC-AUC)
   - **Decision Tree Classifier** (Interpretable tree-based model — 84.9% Accuracy)
2. **Interactive Clinical Streamlit Dashboard**:
   - Input patient demographic, laboratory, and treadmill stress test metrics via interactive sliders.
   - Real-time probabilistic risk prediction with confidence breakdowns.
   - Sample clinical test cases ("🔴 High Risk Case" and "🟢 Healthy Case").
3. **Exploratory Data Analysis (EDA)**:
   - Visualizations displaying Age distribution, Chest Pain correlations, and Maximum Heart Rate vs. Disease status.
4. **Clinical Model Evaluation**:
   - Confusion Matrix with True/False Positive & Negative analysis.
   - Gini Feature Importance chart demonstrating model interpretability.
   - Prioritization of **Recall (Sensitivity)** to minimize critical False Negatives.

---

## 📂 Project Structure

```
heart_disease_predictor/
│
├── data/
│   └── heart.csv              # Heart Disease clinical dataset (5,000 records, 14 features)
├── model/
│   ├── rf_model.pkl           # Trained Random Forest model
│   ├── lr_model.pkl           # Trained Logistic Regression model
│   ├── dt_model.pkl           # Trained Decision Tree model
│   ├── scaler.pkl             # StandardScaler for feature normalization
│   └── metrics.json           # Evaluation metrics, confusion matrix, feature importance
│
├── app.py                     # Streamlit web application
├── CardioGuard_AI.ipynb       # Executable Jupyter Notebook with full analysis & pseudo-code
├── train_model.py             # Model training, evaluation, and serialization script
├── download_data.py           # Dataset acquisition and preprocessing
├── requirements.txt           # Python package dependencies
├── run.bat                    # One-click Windows runner
├── .gitignore                 # Git ignore rules
└── README.md                  # Complete documentation
```

---

## 🚀 How to Run Locally

### Method 1: One-Click Execution (Windows)
Double-click `run.bat` in the project root.

### Method 2: Manual Terminal Commands
1. Clone or open the project folder in terminal:
   ```bash
   cd heart_disease_predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Re-train the models:
   ```bash
   python train_model.py
   ```
4. Launch the Streamlit web dashboard:
   ```bash
   streamlit run app.py
   ```
5. The application will launch at `http://localhost:8501`.

---

## 🌐 How to Deploy to Streamlit Community Cloud (Free & Recommended)

Streamlit Community Cloud provides 100% free hosting directly integrated with your GitHub repository:

1. **Push this repository to GitHub** (see instructions below).
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.
3. Click **"New app"**.
4. Configure the deployment settings:
   - **Repository:** `your-username/heart_disease_predictor`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**.
6. Your web application will be live in ~2 minutes with a shareable public URL (e.g., `https://cardioguard-ai.streamlit.app`).

---

## 🌐 Alternative: Deploy on Render

1. Go to **[render.com](https://render.com/)** and sign in.
2. Click **"New +"** -> **"Web Service"**.
3. Connect your GitHub repository.
4. Configure:
   - **Name:** `cardioguard-heart-disease-predictor`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **"Create Web Service"**.

---

## 👥 Authors & Academic Affiliation

* **Rajat Kumar Sharma** (Roll No: 240101247)
* **Vishal Kumar** (Roll No: 240101248)
* **Ashu Rajpoot** (Roll No: 240101239)

**Course:** B.Tech Computer Science & Engineering — Machine Learning (CS314)  
**Institution:** Indian Institute of Information Technology (IIIT) Bhagalpur  
**Lab Guide:** Dr. Thejaswini M, Assistant Professor, Dept. of CSE
