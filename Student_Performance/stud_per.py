import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_model():
    with open("Student_Performance/strudent_performance_lr_finalmodel.pkl","rb") as file:
        model,scaler,le = pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data,scaler,le):
    data["Extracurricular Activities"]=le.transform([data["Extracurricular Activities"]])[0]
    df = pd.DataFrame([data])
    df_transformed=scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le=load_model()
    preprocessed_data = preprocessing_input_data(data,scaler,le)
    prediction=model.predict(preprocessed_data)
    return prediction

def main():
    st.title("StudentPerformance Prediction")
    st.write("Enter your data to get a prediction")

    hour_studied=st.number_input("Hours Studied",min_value=1,max_value=10,value=5)
    prev_score=st.number_input("Previous Scores",min_value=0,max_value=100,value=60)
    activities=st.selectbox("Extracurricular Activities",['Yes','No'])
    sleep_hours=st.number_input("Sleep Hours",min_value=1,max_value=10,value=5)
    num_question_solved=st.number_input("Sample Question Papers Practiced",min_value=1,max_value=10,value=5)

    if st.button("Predict Your Score"):
        user_data={
            "Hours Studied":hour_studied,
            "Previous Scores":prev_score,
            "Extracurricular Activities":activities,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced":num_question_solved
        }
        prediction=predict_data(user_data)
        st.success(f"Your Prediction is : {prediction}")


if __name__=="__main__":
    main()
