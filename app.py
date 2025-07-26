import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# -------------------------
# 1. Prepare the dataset
# -------------------------
data = {
    'StudyHours': [1, 2, 1.5, 3, 5, 6, 1, 4, 2.5, 3.5, 5.5, 6.5, 0.5, 3, 4.5],
    'Attendance': [60, 65, 55, 75, 90, 95, 50, 85, 70, 80, 92, 96, 45, 78, 88],
    'Pass':       [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df[['StudyHours', 'Attendance']]
y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'student_pass_predictor_model.pkl')

# -------------------------
# 2. Streamlit UI
# -------------------------
st.title("📘 Student Pass/Fail Predictor")
st.write("Enter student details and click **Predict** to see the result.")

# Input fields
study_hours = st.slider("📚 Study Hours per Day", min_value=1, max_value=15, value=5)
attendance = st.slider("📅 Attendance (%)", min_value=0, max_value=100, value=75)

# Prediction Button
if st.button("🔮 Predict"):
    input_data = pd.DataFrame([[study_hours, attendance]], columns=['StudyHours', 'Attendance'])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display Result
    if prediction == 1:
        st.success(f"✅ Prediction: The student is likely to PASS (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Prediction: The student is likely to FAIL (Confidence: {1 - probability:.2f})")
# Save the model for future use
import joblib
joblib.dump(model, 'student_pass_predictor_model.pkl')
# Note: The model can be loaded later using joblib.load('student_pass_predictor_model.pkl')
# This code creates a simple Streamlit app to predict whether a student will pass or fail based on their study hours and attendance.
# The model is trained on a small dataset and can be improved with more data.   
# To run the app, save this code in a file (e.g., student_pass_predictor.py) and run it using the command:
# streamlit run student_pass_predictor.py   