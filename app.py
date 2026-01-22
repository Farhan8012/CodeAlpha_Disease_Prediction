import streamlit as st
import pickle
import numpy as np

# 1. Load the saved model and scaler
model = pickle.load(open('svm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# 2. Set the title and description
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict heart disease risk.")

# 3. Create Input Fields for the user
# We need inputs for all 13 features expected by the model

col1, col2, col3 = st.columns(3) # Split screen into 3 columns for better look

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], help="1: Typical Angina, 2: Atypical, 3: Non-anginal, 4: Asymptomatic")
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["False", "True"])
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina?", options=["No", "Yes"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0)

with col3:
    slope = st.selectbox("Slope of Peak Exercise ST", options=[1, 2, 3])
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}[x])

# 4. Convert text inputs to numbers (Preprocessing)
# The model only understands numbers, so we convert "Male" -> 1, "Female" -> 0, etc.

sex_num = 1 if sex == "Male" else 0
fbs_num = 1 if fbs == "True" else 0
exang_num = 1 if exang == "Yes" else 0

# 5. Create the prediction button
if st.button("Predict"):
    # Combine all inputs into a single list
    user_input = [age, sex_num, cp, trestbps, chol, fbs_num, restecg, thalach, exang_num, oldpeak, slope, ca, thal]
    
    # Scale the input (just like we did in training!)
    # Reshape because scaler expects a 2D array
    user_input_scaled = scaler.transform([user_input])
    
    # Make Prediction
    prediction = model.predict(user_input_scaled)
    
    # Display Result
    if prediction[0] == 1:
        st.error("⚠️ Prediction: HIGH RISK of Heart Disease")
        st.write("Please consult a cardiologist immediately.")
    else:
        st.success("✅ Prediction: HEALTHY Heart")
        st.write("No signs of heart disease detected.")