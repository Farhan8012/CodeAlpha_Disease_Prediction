# Disease Prediction System (Heart Disease)

## Overview
This project predicts the likelihood of heart disease in patients based on medical attributes such as age, cholesterol levels, and blood pressure. It uses Machine Learning algorithms and a Streamlit web interface for easy user interaction.

graph TD
    subgraph Data Pipeline
        A[UCI Heart Disease Dataset] -->|Load Data| B(Data Cleaning & Preprocessing);
        B -->|Encoding & Scaling| C{Train/Test Split};
        C -- Training Data --> D[Train Models: LR, SVM, RF, XGBoost];
        D --> E(Compare Model Accuracy);
    end

    subgraph Model Selection
        E --> F{Select Best Model: SVM ~90%};
        F -- Save Artifacts --> G[(svm_model.pkl & scaler.pkl)];
    end

    subgraph Streamlit Application
        H[Web Interface User Inputs] --> I(Scale New Data);
        G -.->|Load Saved Model| J(Make Prediction);
        I --> J;
        J --> K{Prediction Result};
        K -- 0 --> L[✅ HEALTHY Heart];
        K -- 1 --> M[⚠️ HIGH RISK];
    end

    style F fill:#d4edda,stroke:#28a745,stroke-width:2px
    style L fill:#d4edda,stroke:#28a745,color:#155724
    style M fill:#f8d7da,stroke:#dc3545,color:#721c24

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4f77481d-18e2-499b-a7c3-dea24e333cee" />


## Features
* **Data Analysis:** Exploratory Data Analysis (EDA) with correlation heatmaps.
* **Model Comparison:** Compared Logistic Regression, SVM, Random Forest, and XGBoost.
* **Best Model:** Support Vector Machine (SVM) achieved ~90% accuracy.
* **Web App:** Interactive dashboard built with Streamlit.

## Tech Stack
* Python
* Scikit-Learn, XGBoost
* Pandas, NumPy
* Streamlit (for the interface)

## How to Run locally
1. Clone the repository:
   git clone [https://github.com/Farhan8012/CodeAlpha_Disease_Prediction.git](https://github.com/Farhan8012/CodeAlpha_Disease_Prediction.git)

2. Install dependencies:
   pip install pandas numpy scikit-learn xgboost streamlit ucimlrepo matplotlib seaborn

3. Run the application:
   streamlit run app.py
