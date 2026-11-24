import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PDF = os.path.join(BASE_DIR, "Project_Presentation_Guide.pdf")
UPDATED_PDF = os.path.join(BASE_DIR, "Project_Presentation_Guide_Updated.pdf")
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

def create_page(fig, title_text, subtitle_text, page_num):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Header Background Bar
    header_box = patches.FancyBboxPatch(
        (0.04, 0.90), 0.92, 0.08,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        facecolor="#0f172a", edgecolor="none", zorder=1
    )
    ax.add_patch(header_box)
    
    # Red accent tag in header
    accent_box = patches.FancyBboxPatch(
        (0.06, 0.952), 0.24, 0.018,
        boxstyle="round,pad=0.005,rounding_size=0.005",
        facecolor="#ef4444", edgecolor="none", zorder=2
    )
    ax.add_patch(accent_box)
    ax.text(0.18, 0.961, "CARDIOVASCULAR AI SYSTEM", ha='center', va='center',
            color='#ffffff', fontsize=7.5, weight='bold', zorder=3)
    
    # Title & Subtitle text
    ax.text(0.06, 0.932, title_text, ha='left', va='center',
            color='#ffffff', fontsize=13.5, weight='bold', zorder=2)
    ax.text(0.06, 0.912, subtitle_text, ha='left', va='center',
            color='#94a3b8', fontsize=8.5, zorder=2)
    
    # Footer
    ax.plot([0.04, 0.96], [0.035, 0.035], color='#cbd5e1', lw=0.8)
    ax.text(0.04, 0.022, "CardioGuard Presentation Guide | Academic & Clinical Decision Support System",
            color='#64748b', fontsize=7.5, va='center')
    ax.text(0.96, 0.022, f"Page {page_num} of 5",
            color='#64748b', fontsize=7.5, ha='right', va='center')
    
    return ax

def draw_card(ax, x, y, w, h, bg_color="#f8fafc", border_color="#e2e8f0", radius=0.012):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=bg_color, edgecolor=border_color, lw=1.2, zorder=1
    )
    ax.add_patch(box)

def generate_guide():
    # Load dataset & metrics dynamically
    df = pd.read_csv(DATA_PATH)
    n_total = len(df)
    n_pos = int((df['target'] == 1).sum())
    n_neg = int((df['target'] == 0).sum())
    pct_pos = (n_pos / n_total) * 100
    pct_neg = (n_neg / n_total) * 100
    n_train = int(n_total * 0.8)
    n_test = n_total - n_train

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    rf_m = metrics["models"]["Random Forest"]
    lr_m = metrics["models"]["Logistic Regression"]
    dt_m = metrics["models"]["Decision Tree"]

    cm = rf_m["confusion_matrix"]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    test_sick = tp + fn
    test_healthy = tn + fp

    target_pdf = None
    for candidate in [OUTPUT_PDF, UPDATED_PDF, os.path.join(BASE_DIR, "Project_Presentation_Guide_v3.pdf")]:
        try:
            with open(candidate, 'a'):
                pass
            target_pdf = candidate
            break
        except PermissionError:
            continue

    if not target_pdf:
        target_pdf = os.path.join(BASE_DIR, "Project_Presentation_Guide_v3.pdf")

    with PdfPages(target_pdf) as pdf:
        
        # ==========================================
        # PAGE 1: INTRODUCTION & 60-SEC ELEVATOR PITCH
        # ==========================================
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        ax = create_page(fig, "1. Project Introduction & 60-Second Speaking Pitch", 
                         "How to start your presentation when Mam asks: 'Explain your project'", 1)
        
        # Card 1: 60-Second Elevator Pitch
        draw_card(ax, 0.05, 0.69, 0.90, 0.19, bg_color="#f0fdf4", border_color="#22c55e")
        ax.text(0.07, 0.86, ">> WHAT EXACTLY TO SAY TO MAM (Your 60-Second Opening Pitch):", 
                fontsize=9.5, weight='bold', color="#166534")
        
        pitch_script = (
            "\"Good morning / afternoon Mam.\n\n"
            "My project is titled 'CardioGuard: Clinical Heart Disease Risk Assessment System'.\n\n"
            "Cardiovascular disease is the number one cause of mortality worldwide. The main challenge in "
            "clinical practice is that early symptoms are often silent or misdiagnosed.\n\n"
            "In this project, I developed an intelligent clinical decision-support system using Supervised "
            "Machine Learning. We trained and evaluated multiple classification algorithms — including "
            "Random Forest, Logistic Regression, and Decision Trees — on an expanded cohort of 5,000 standardized "
            "clinical patient records.\n\n"
            "Our primary objective was not just high accuracy, but maximizing RECALL (Sensitivity), because "
            "in healthcare, missing a sick patient (a False Negative) is fatal. Finally, I deployed the complete system "
            "as an interactive Streamlit web application providing instant patient stratification and explainable AI insights.\""
        )
        ax.text(0.07, 0.835, pitch_script, fontsize=8.1, color="#1e293b", va='top', linespacing=1.35)
        
        # Card 2: What problem does it solve?
        draw_card(ax, 0.05, 0.49, 0.90, 0.18, bg_color="#ffffff", border_color="#e2e8f0")
        ax.text(0.07, 0.65, "[THE CLINICAL PROBLEM WE ARE SOLVING]", fontsize=9.5, weight='bold', color="#0f172a")
        
        prob_text = (
            "1. Diagnostic Delays: Traditional coronary angiography is invasive, costly, and requires hospitalization.\n"
            "2. High Rate of Missed Cases: Subtle ischemic changes on treadmill stress tests or asymptomatic angina\n"
            "   are frequently overlooked during manual triage.\n"
            "3. Need for Probabilistic Risk Scoring: Doctors do not just want a binary Yes/No; they need a calibrated\n"
            "   percentage probability score (e.g., 'Patient has an 87.4% risk of coronary stenosis') to decide whether\n"
            "   to order urgent catheterization or prescribe preventive statins."
        )
        ax.text(0.07, 0.625, prob_text, fontsize=8.2, color="#334155", va='top', linespacing=1.4)
        
        # Card 3: Project Objectives
        draw_card(ax, 0.05, 0.29, 0.90, 0.18, bg_color="#ffffff", border_color="#e2e8f0")
        ax.text(0.07, 0.45, "[CORE PROJECT OBJECTIVES]", fontsize=9.5, weight='bold', color="#0f172a")
        
        obj_text = (
            "1. Multi-Model Benchmark: Empirically compare linear, tree-based, and ensemble ML classifiers.\n"
            "2. Clinical Optimization for Recall: Minimize False Negatives (Type II Errors) to safeguard patient survival.\n"
            "3. Explainability & Interpretability: Compute feature importance scores so doctors know WHY a prediction\n"
            "   was made (Gini Impurity reduction).\n"
            "4. End-to-End Practical Implementation: Implement a complete automated pipeline from data preprocessing\n"
            "   and standard scaling to interactive Streamlit GUI deployment."
        )
        ax.text(0.07, 0.425, obj_text, fontsize=8.2, color="#334155", va='top', linespacing=1.4)
        
        # Card 4: Quick Summary Table Box
        draw_card(ax, 0.05, 0.07, 0.90, 0.20, bg_color="#f8fafc", border_color="#cbd5e1")
        ax.text(0.07, 0.25, "[PROJECT SPECIFICATIONS AT A GLANCE]", fontsize=9.5, weight='bold', color="#0f172a")
        
        specs_text = (
            f"- Programming Language: Python 3.12\n"
            f"- Core Libraries: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, Joblib\n"
            f"- Dataset: Expanded Clinical Cardiovascular Cohort (UCI Cleveland & Framingham Clinical Schema)\n"
            f"- Total Dataset Dimensions: {n_total:,} Rows (Patients) x 14 Columns (13 Input Features + 1 Output Target)\n"
            f"- Output Target Variable: Binary classification (1 = Disease Present [{n_pos:,} rows, {pct_pos:.1f}%], 0 = Healthy [{n_neg:,} rows, {pct_neg:.1f}%])\n"
            f"- Best Selected Model: Random Forest (Accuracy: {rf_m['accuracy']}%, Recall: {rf_m['recall']}%, ROC-AUC: {rf_m['roc_auc']}%)\n"
            f"- User Interface: Local Streamlit web dashboard with interactive sliders and instant case presets."
        )
        ax.text(0.07, 0.225, specs_text, fontsize=8.2, color="#1e293b", va='top', linespacing=1.38)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==========================================
        # PAGE 2: DATASET & INPUT/OUTPUT FEATURES
        # ==========================================
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        ax = create_page(fig, "2. Dataset, Input Features & Output Target Explained", 
                         "Dataset dimensions, 13 input clinical features, and 1 binary output variable", 2)
        
        # Card 1: Dataset Overview & Dimensions
        draw_card(ax, 0.05, 0.765, 0.90, 0.12, bg_color="#f1f5f9", border_color="#cbd5e1")
        ax.text(0.07, 0.865, "[DATASET NAME, ROWS & COLUMNS BREAKDOWN]", fontsize=9.5, weight='bold', color="#0f172a")
        ds_info = (
            f"- Dataset: Expanded Clinical Cardiovascular Cohort (Cleveland & Framingham Clinical Schema)\n"
            f"- Total Rows (Patient Records): {n_total:,} ROWS (Statistically robust sample size with zero nulls)\n"
            f"- Total Columns: 14 COLUMNS = 13 INPUT FEATURES + 1 OUTPUT TARGET FEATURE\n"
            f"- Data Split: 80% Training ({n_train:,} rows) and 20% Testing ({n_test:,} rows) using Stratified Sampling"
        )
        ax.text(0.07, 0.84, ds_info, fontsize=8.2, color="#1e293b", va='top', linespacing=1.35)
        
        # Card 2: The Output Feature
        draw_card(ax, 0.05, 0.615, 0.90, 0.135, bg_color="#eff6ff", border_color="#3b82f6")
        ax.text(0.07, 0.73, "[THE 1 OUTPUT FEATURE (TARGET VARIABLE) EXPLAINED]", fontsize=9.5, weight='bold', color="#1d4ed8")
        target_info = (
            f"• Column Name: `target` (Integer binary: 0 or 1)\n"
            f"• Medical Definition: Indicates whether coronary angiography detected >= 50% diameter narrowing\n"
            f"  in at least one of the patient's major coronary arteries.\n"
            f"  - Value `1` = Heart Disease Present (Positive Case) -> {n_pos:,} Rows ({pct_pos:.1f}% of dataset)\n"
            f"  - Value `0` = Healthy / Normal Baseline (Negative Case) -> {n_neg:,} Rows ({pct_neg:.1f}% of dataset)\n"
            f"• In the Streamlit UI, the model predicts the class and outputs exact probabilistic risk confidence (0-100%)."
        )
        ax.text(0.07, 0.705, target_info, fontsize=8.1, color="#1e3a8a", va='top', linespacing=1.3)

        # Card 3: The 13 Input Features Table (Divided into 3 Clinical Categories)
        draw_card(ax, 0.05, 0.055, 0.90, 0.545, bg_color="#ffffff", border_color="#e2e8f0")
        ax.text(0.07, 0.582, "[THE 13 INPUT FEATURES (EXPLAINED IN SIMPLE WORDS FOR MAM)]", fontsize=9.5, weight='bold', color="#0f172a")
        
        feat_content = (
            "CATEGORY A: DEMOGRAPHICS & RESTING VITALS (4 Features)\n"
            "  1. `age` (Age): Patient age in years. Older patients have higher arterial stiffness and plaque.\n"
            "  2. `sex` (Sex): Biological gender (1 = Male, 0 = Female). Males statistically show earlier CAD onset.\n"
            "  3. `trestbps` (Resting BP): Resting blood pressure in mm Hg on admission (Normal: < 120 mm Hg).\n"
            "  4. `chol` (Cholesterol): Serum cholesterol in mg/dl (Desirable: < 200 mg/dl; excess causes plaque).\n\n"
            "CATEGORY B: LAB METRICS & RESTING ECG (4 Features)\n"
            "  5. `fbs` (Fasting Blood Sugar): Fasting blood sugar > 120 mg/dl (1 = Elevated/Diabetic, 0 = Normal).\n"
            "  6. `cp` (Chest Pain Type): Categorized into 4 clinical types (Strongest clinical indicator):\n"
            "     - Type 0: Typical Angina (Classic retrosternal squeezing pain radiating to left arm/jaw)\n"
            "     - Type 1: Atypical Angina (Dyspnea, shortness of breath, epigastric discomfort)\n"
            "     - Type 2: Non-anginal Pain (Superficial musculoskeletal or GI pain unrelated to ischemia)\n"
            "     - Type 3: Asymptomatic (Silent ischemia, frequently observed in diabetic neuropathy)\n"
            "  7. `restecg` (Resting ECG): 0 = Normal, 1 = ST-T wave abnormalities, 2 = Left Ventricular Hypertrophy.\n"
            "  8. `thalach` (Max Heart Rate): Maximum heart rate achieved during treadmill stress test (bpm).\n"
            "     (Healthy hearts reach high peak rates; diseased hearts suffer from cardiac chronotropic failure).\n\n"
            "CATEGORY C: TREADMILL STRESS TEST & ANGIOGRAPHY (5 Features)\n"
            "  9. `exang` (Exercise Angina): Angina provoked by physical exertion (1 = Yes, 0 = No).\n"
            "  10. `oldpeak` (ST Depression): ST-segment depression induced by exercise relative to rest (in mm).\n"
            "      (Deep depression > 1.5mm indicates severe subendocardial ischemia under workload).\n"
            "  11. `slope` (ST Segment Slope): Slope of peak exercise ST segment (0 = Upsloping, 1 = Flat, 2 = Downsloping).\n"
            "  12. `ca` (Major Vessels): Number of major coronary arteries (0-3) seen under fluoroscopy.\n"
            "      (Blocked arteries fail to transmit radiopaque dye; 0 = clear flow, 1-3 = blocked vessels).\n"
            "  13. `thal` (Thallium Scintigraphy): 1 = Normal, 2 = Fixed defect (prior infarct), 3 = Reversible defect."
        )
        ax.text(0.07, 0.558, feat_content, fontsize=7.8, color="#1e293b", va='top', linespacing=1.24)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==========================================
        # PAGE 3: ML WORKFLOW & RECALL CONCEPT
        # ==========================================
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        ax = create_page(fig, "3. ML Pipeline & The 'Recall' Justification", 
                         "How models were trained and why Recall is the #1 metric in healthcare", 3)
        
        # Card 1: Workflow Steps
        draw_card(ax, 0.05, 0.69, 0.90, 0.19, bg_color="#ffffff", border_color="#e2e8f0")
        ax.text(0.07, 0.86, "[END-TO-END MACHINE LEARNING PIPELINE]", fontsize=9.5, weight='bold', color="#0f172a")
        pipe_text = (
            f"Step 1: Data Ingestion & Integrity Check -> Loaded {n_total:,} rows; verified zero null/missing values.\n"
            f"Step 2: Train-Test Partitioning -> 80% train ({n_train:,} rows), 20% test ({n_test:,} rows) using Stratified split.\n"
            f"Step 3: Feature Scaling -> Applied StandardScaler (Z-score normalization) for Logistic Regression\n"
            f"        because distance-based gradient optimization requires features to share zero mean and unit variance.\n"
            f"Step 4: Multi-Model Training -> Trained Random Forest (Ensemble), Logistic Regression, Decision Tree.\n"
            f"Step 5: Quantitative Evaluation -> Generated Confusion Matrices, Precision, Recall, F1, and ROC-AUC curves.\n"
            f"Step 6: Serialization & Deployment -> Models saved as .pkl via Joblib and loaded into interactive Streamlit UI."
        )
        ax.text(0.07, 0.835, pipe_text, fontsize=8.2, color="#334155", va='top', linespacing=1.35)
        
        # Card 2: Model Benchmark Table
        draw_card(ax, 0.05, 0.44, 0.90, 0.23, bg_color="#f8fafc", border_color="#cbd5e1")
        ax.text(0.07, 0.65, "[MODEL BENCHMARK COMPARISON TABLE]", fontsize=9.5, weight='bold', color="#0f172a")
        
        table_text = (
            "Model Name                Accuracy    Recall (Sensitivity)    Precision    F1-Score    ROC-AUC\n"
            "----------------------------------------------------------------------------------------------------\n"
            f"Random Forest (Best)       {rf_m['accuracy']:.2f}%      {rf_m['recall']:.2f}% [PRIMARY]       {rf_m['precision']:.2f}%       {rf_m['f1_score']:.2f}%      {rf_m['roc_auc']:.2f}%\n"
            f"Logistic Regression        {lr_m['accuracy']:.2f}%      {lr_m['recall']:.2f}%                 {lr_m['precision']:.2f}%       {lr_m['f1_score']:.2f}%      {lr_m['roc_auc']:.2f}%\n"
            f"Decision Tree              {dt_m['accuracy']:.2f}%      {dt_m['recall']:.2f}%                 {dt_m['precision']:.2f}%       {dt_m['f1_score']:.2f}%      {dt_m['roc_auc']:.2f}%\n"
            "----------------------------------------------------------------------------------------------------\n\n"
            "Key Observation to tell Mam:\n"
            "Random Forest is selected as primary because it combines 100 individual decision trees via Bootstrap\n"
            "Aggregation (Bagging) and random subspace feature sampling. This dramatically reduces variance and maximizes\n"
            "clinical Recall (Sensitivity) to reliably protect patient lives."
        )
        ax.text(0.07, 0.625, table_text, fontsize=8.0, color="#0f172a", va='top', linespacing=1.3, fontfamily='monospace')
        
        # Card 3: The Golden Concept (Recall vs Accuracy)
        draw_card(ax, 0.05, 0.06, 0.90, 0.36, bg_color="#eff6ff", border_color="#3b82f6")
        ax.text(0.07, 0.40, "[THE #1 QUESTION: WHY IS RECALL MORE IMPORTANT THAN ACCURACY?]", 
                fontsize=9.5, weight='bold', color="#1d4ed8")
        
        recall_explanation = (
            "If Mam asks: 'Why do you care so much about Recall instead of just Accuracy?', answer this:\n\n"
            "1. The Two Types of Errors in Medical ML:\n"
            "   - Type I Error (False Positive): A healthy person is predicted to have heart disease.\n"
            "     -> Consequence: Safe. The patient undergoes a confirmatory ECG or blood test, and the doctor\n"
            "        realizes they are healthy. Mild inconvenience, but ZERO risk to human life.\n\n"
            "   - Type II Error (False Negative): A severely diseased patient is predicted to be HEALTHY.\n"
            "     -> Consequence: FATAL. The patient is sent home without medication, angioplasty, or monitoring.\n"
            "        They may suffer a fatal myocardial infarction (heart attack) within days.\n\n"
            "2. Mathematical Definition of Recall:\n"
            "              Recall = True Positives / (True Positives + False Negatives)\n"
            "   To make False Negatives approach zero, Recall MUST approach 100%.\n\n"
            "3. Conclusion for Mam:\n"
            f"   'Therefore Mam, our Random Forest achieves {rf_m['recall']:.2f}% Recall (catching {tp:,} out of {test_sick:,} sick patients)\n"
            f"   and {rf_m['roc_auc']:.2f}% ROC-AUC across {n_test:,} held-out test patients!'"
        )
        ax.text(0.07, 0.375, recall_explanation, fontsize=8.1, color="#1e3a8a", va='top', linespacing=1.32)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==========================================
        # PAGE 4: STREAMLIT APP WALKTHROUGH SCRIPT
        # ==========================================
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        ax = create_page(fig, "4. Step-by-Step Live UI Demonstration Script", 
                         "Exactly what to click, demonstrate, and explain during the live software demo", 4)
        
        # Demo Step 1: Sidebar
        draw_card(ax, 0.05, 0.74, 0.90, 0.145, bg_color="#ffffff", border_color="#e2e8f0")
        ax.text(0.07, 0.86, "STEP 1: LAUNCH & SIDEBAR OVERVIEW", fontsize=9.5, weight='bold', color="#0f172a")
        s1 = (
            "- What to do: Open terminal and type `streamlit run app.py` (or double-click `run.bat`).\n"
            "- What to say: 'Mam, here is our live application, CardioGuard, running on a local Streamlit web server.\n"
            f"  The sidebar contains algorithm selection, instant test case profiles, and dataset metadata ({n_total:,} records).'\n"
            "- Point out: 'Notice in the sidebar that Random Forest is pre-selected as our recommended primary model.'"
        )
        ax.text(0.07, 0.835, s1, fontsize=8.2, color="#334155", va='top', linespacing=1.35)
        
        # Demo Step 2: Tab 1 (Risk Assessment)
        draw_card(ax, 0.05, 0.50, 0.90, 0.22, bg_color="#ffffff", border_color="#e2e8f0")
        ax.text(0.07, 0.695, "STEP 2: TAB 1 - LIVE PATIENT RISK PREDICTION", fontsize=9.5, weight='bold', color="#0f172a")
        s2 = (
            "- Action A: Click the '🔴 High Risk Case' quick preset in the sidebar.\n"
            "  * Sliders instantly fill with high-risk clinical markers (Oldpeak = 3.2mm, Vessel Ca = 2, Age = 64).\n"
            "  * Click 'Run Clinical Risk Assessment' button.\n"
            "  * What to say: 'As expected, the model triggers a High Risk alert with over 90% confidence, advising\n"
            "    urgent coronary angiography and cardiologist referral.'\n\n"
            "- Action B: Click the '🟢 Healthy Case' quick preset in the sidebar.\n"
            "  * Sliders instantly update (Oldpeak = 0.2mm, Normal ECG, Age = 45).\n"
            "  * Click the button again.\n"
            "  * What to say: 'Now the model outputs a Healthy Baseline status with green low-risk indicators.'"
        )
        ax.text(0.07, 0.67, s2, fontsize=8.1, color="#334155", va='top', linespacing=1.3)
        
        # Demo Step 3: Tab 2 (EDA)
        draw_card(ax, 0.05, 0.26, 0.90, 0.22, bg_color="#ffffff", border_color="#e2e8f0")
        ax.text(0.07, 0.455, "STEP 3: TAB 2 - CLINICAL EXPLORATORY DATA ANALYSIS (EDA)", fontsize=9.5, weight='bold', color="#0f172a")
        s3 = (
            "- Click on Tab 2: 'Clinical Exploratory Data Analysis'.\n"
            "- Explain the 4 graphs briefly to Mam:\n"
            "  1. Age Distribution: 'Patients aged 50-65 have significantly higher rates of coronary disease.'\n"
            "  2. Peak Heart Rate vs Age: 'Healthy patients maintain higher maximum heart rates under physical stress,\n"
            "     while diseased patients suffer from cardiac chronotropic incompetence.'\n"
            "  3. Chest Pain Severity: 'Shows clear statistical link between angina symptoms and positive target labels.'\n"
            "  4. Correlation Heatmap: 'Shows oldpeak and chest pain have the strongest statistical coefficients with target.'"
        )
        ax.text(0.07, 0.415, s3, fontsize=8.1, color="#334155", va='top', linespacing=1.3)
        
        # Demo Step 4: Tab 3 (Model Performance)
        draw_card(ax, 0.05, 0.06, 0.90, 0.18, bg_color="#ffffff", border_color="#e2e8f0")
        ax.text(0.07, 0.22, "STEP 4: TAB 3 - MODEL BENCHMARK & EVALUATION", fontsize=9.5, weight='bold', color="#0f172a")
        s4 = (
            "- Click on Tab 3: 'Model Benchmark & Evaluation'.\n"
            "- Point out the Confusion Matrix:\n"
            f"  * 'Mam, out of {test_sick:,} diseased patients in our {n_test:,}-case test set, the model correctly identified {tp:,} cases (True Positives),\n"
            f"    resulting in a {rf_m['recall']:.2f}% Recall rate with only {fn} False Negatives (Overall Accuracy: {rf_m['accuracy']:.2f}%).'\n"
            "- Point out the Feature Importance chart:\n"
            "  * 'This Gini importance chart proves that Chest Pain (`cp`), ST Depression (`oldpeak`), and Max Heart Rate\n"
            "    (`thalach`) are the strongest mathematical drivers in Random Forest decision splits.'"
        )
        ax.text(0.07, 0.195, s4, fontsize=8.1, color="#334155", va='top', linespacing=1.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ==========================================
        # PAGE 5: TOP QUESTIONS MAM WILL ASK
        # ==========================================
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        ax = create_page(fig, "5. Top Questions Mam Might Ask & Simple English Answers", 
                         "High-confidence answers for common academic & technical inquiries", 5)
        
        # Q1 & Q2
        draw_card(ax, 0.05, 0.65, 0.90, 0.23, bg_color="#ffffff", border_color="#e2e8f0")
        ax.text(0.07, 0.86, "Q1: Why did you scale features for Logistic Regression but not Random Forest?", 
                fontsize=8.8, weight='bold', color="#0f172a")
        a1 = (
            "Answer: Logistic Regression is a parametric model that uses gradient optimization and regularization (L1/L2).\n"
            "If features have drastically different ranges (e.g. Cholesterol: 126-564 vs Oldpeak: 0-6), the gradient updates\n"
            "get biased toward larger scales. Scaling brings all features to mean 0 and variance 1.\n"
            "Decision Trees and Random Forests, however, are non-parametric and scale-invariant because splits are based solely\n"
            "on monotonic rank order (e.g. is X > threshold?), so distance scaling does not affect splits."
        )
        ax.text(0.07, 0.835, a1, fontsize=7.9, color="#334155", va='top', linespacing=1.3)
        
        ax.text(0.07, 0.73, "Q2: What is the difference between Decision Tree and Random Forest?", 
                fontsize=8.8, weight='bold', color="#0f172a")
        a2 = (
            "Answer: A single Decision Tree learns hierarchical if-else decision rules. While easy to visualize, single trees\n"
            "suffer from high variance and easily overfit noisy training samples.\n"
            "Random Forest is an ensemble technique: it creates an entire forest (100 trees) using Bagging (Bootstrap Aggregation)\n"
            "and random feature subsets. Each tree votes, and the majority vote decides the output. This reduces overall variance\n"
            "and produces much higher generalization accuracy."
        )
        ax.text(0.07, 0.705, a2, fontsize=7.9, color="#334155", va='top', linespacing=1.3)
        
        # Q3 & Q4
        draw_card(ax, 0.05, 0.35, 0.90, 0.28, bg_color="#ffffff", border_color="#e2e8f0")
        ax.text(0.07, 0.61, "Q3: What does the Confusion Matrix show?", 
                fontsize=8.8, weight='bold', color="#0f172a")
        a3 = (
            "Answer: The confusion matrix summarizes the exact classification counts on test data:\n"
            "- True Negatives (TN): Healthy individuals correctly classified as healthy.\n"
            "- True Positives (TP): Diseased patients correctly identified as having heart disease.\n"
            "- False Positives (FP): Healthy patients falsely flagged as sick (Type I Error).\n"
            "- False Negatives (FN): Diseased patients missed and falsely labelled healthy (Type II Error - highly dangerous!)."
        )
        ax.text(0.07, 0.585, a3, fontsize=7.9, color="#334155", va='top', linespacing=1.3)
        
        ax.text(0.07, 0.47, "Q4: How does the model output a probability percentage?", 
                fontsize=8.8, weight='bold', color="#0f172a")
        a4 = (
            "Answer: In Random Forest, `predict_proba()` calculates the proportion of individual decision trees that voted\n"
            "for Class 1 (Disease). For example, if 88 out of 100 trees vote 'Disease', the probability is 88.0%.\n"
            "In Logistic Regression, the probability is computed mathematically using the Sigmoid (logistic) function:\n"
            "P(Y=1|X) = 1 / (1 + e^(-z)), where z is the linear combination of weighted features."
        )
        ax.text(0.07, 0.445, a4, fontsize=7.9, color="#334155", va='top', linespacing=1.3)
        
        # Q5 & Future Scope
        draw_card(ax, 0.05, 0.06, 0.90, 0.27, bg_color="#f8fafc", border_color="#cbd5e1")
        ax.text(0.07, 0.31, "Q5: What are the limitations and future enhancements of this project?", 
                fontsize=8.8, weight='bold', color="#0f172a")
        a5 = (
            f"Answer: Current Limitations & Scope:\n"
            f"1. Dataset Scale: Evaluated on an expanded {n_total:,}-patient clinical cohort with {n_test:,} held-out test evaluations.\n"
            f"   In real multi-national production, systems integrate with millions of multi-hospital records.\n"
            f"2. Static Features: The model takes numerical clinical lab values, not continuous live biometric waveforms.\n\n"
            f"Future Scope (To impress Mam):\n"
            f"- Integration with IoT Wearables: Ingest live smartwatch photoplethysmography (PPG) and ECG telemetry.\n"
            f"- Deep Learning Vision: Combine clinical tabular data with CNN models trained on chest X-rays and angiograms.\n"
            f"- Hospital EHR Integration: Connect via HL7 / FHIR protocols to automatically auto-fill lab values from hospital servers."
        )
        ax.text(0.07, 0.285, a5, fontsize=7.9, color="#1e293b", va='top', linespacing=1.28)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"PDF successfully generated at: {target_pdf}")

if __name__ == "__main__":
    generate_guide()
