import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import streamlit as st

# Load the CSV file
def load_data(csv_file):
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    else:
        st.write(f"No file found at {csv_file}. Starting with an empty dataset.")
        return pd.DataFrame(columns=['Gender', 'Height', 'Weight', 'Index'])

# Save the data to the CSV file
def save_data(df, csv_file):
    df.to_csv(csv_file, index=False)

# Function to add a new record
def add_record(df, csv_file):
    st.subheader("Add a new record")
    gender = st.radio("Select gender", ('Male', 'Female'))
    height = st.number_input("Enter height in cm", min_value=0)
    weight = st.number_input("Enter weight in kg", min_value=0)
    bmi_index = st.slider("Enter BMI index (0-4)", 0, 4)

    if st.button("Add Record"):
        new_record = pd.DataFrame([[gender, height, weight, bmi_index]], columns=['Gender', 'Height', 'Weight', 'Index'])
        df = pd.concat([df, new_record], ignore_index=True)
        save_data(df, csv_file)
        st.success("Record added successfully.")
    return df

# Function to train the machine learning model
def train_model(df):
    if df.shape[0] < 2:
        st.warning("Not enough data to train the model.")
        return None
    
    # Convert 'Gender' to numerical values
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    X = df[['Gender', 'Height', 'Weight']]
    y = df['Index']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to predict BMI Index
def predict_bmi(model):
    st.subheader("Predict BMI Index")
    gender = st.radio("Select gender", ('Male', 'Female'))
    height = st.number_input("Enter height in cm", min_value=0)
    weight = st.number_input("Enter weight in kg", min_value=0)
    
    if st.button("Predict"):
        gender_numeric = 0 if gender == 'Male' else 1
        predicted_index = model.predict(np.array([[gender_numeric, height, weight]]))
        st.write(f"Predicted BMI Index: {round(predicted_index[0], 2)}")

# Function to view all records
def view_records(df):
    st.subheader("View all records")
    if df.empty:
        st.write("No records found.")
    else:
        st.dataframe(df)

# Main application loop
def main():
    st.title("BMI Tracking App")
    
    csv_file = "bmi.csv"
    df = load_data(csv_file)
    
    menu = ["Add a new record", "Predict BMI Index", "View all records"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == 'Add a new record':
        df = add_record(df, csv_file)
    elif choice == 'Predict BMI Index':
        model = train_model(df)
        if model:
            predict_bmi(model)
    elif choice == 'View all records':
        view_records(df)

if __name__ == "__main__":
    main()
