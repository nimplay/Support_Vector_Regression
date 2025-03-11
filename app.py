import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Salary Prediction App", page_icon="💰", layout="wide")

# Custom styles
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.title("💰 Salary Prediction using Support Vector Regression (SVR)")

# Manual CSV upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully.")
else:
    # Load default data
    data = pd.read_csv('professional_data.csv')
    st.sidebar.info("Using default data: 'professional_data.csv'.")

# Display the dataset in the app
st.subheader("📊 Professional Data")
st.write(data)

# Prepare the data
if 'Experience' in data.columns and 'educational_level' in data.columns and 'skills' in data.columns and 'salary' in data.columns:
    X = data[['Experience', 'educational_level', 'skills']]
    y = data['salary']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the SVR model
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)

    # Prediction section
    st.subheader("🔮 Make a Prediction")

    # User inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        experience = st.number_input("Experience (years)", min_value=0, max_value=30, value=5)
    with col2:
        education_level = st.selectbox("Education Level", options=[1, 2, 3], format_func=lambda x: ["Bachelor's", "Master's", "PhD"][x-1])
    with col3:
        skills = st.number_input("Technical Skills", min_value=1, max_value=10, value=5)

    # Preprocess the input data
    input_data = np.array([[experience, education_level, skills]])
    input_scaled = scaler.transform(input_data)

    # Make a prediction
    if st.button("Predict Salary"):
        prediction = model.predict(input_scaled)
        st.success(f"The predicted salary is: **${prediction[0]:.2f}** thousand dollars.")

        # Plot the actual data vs prediction
        plt.figure(figsize=(10, 6))
        plt.scatter(data['Experience'], data['salary'], color='blue', label='Actual Data')
        plt.scatter(experience, prediction, color='red', label='Prediction', s=100)
        plt.xlabel('Experience (years)')
        plt.ylabel('Salary (thousands of dollars)')
        plt.title('Salary Prediction using SVR')
        plt.legend()
        st.pyplot(plt)

    # Model evaluation section
    st.subheader("📈 Model Evaluation")
    st.write("The model was evaluated on a test set. Here are the results:")

    # Calculate the model's score
    score = model.score(X_test, y_test)
    st.write(f"**Model Score (R²):** {score:.2f}")

    # Additional graphs for model evaluation
    st.subheader("📊 Model Performance Visualizations")

    # Graph 1: Actual vs Predicted Salaries (Test Set)
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Prediction')
    plt.xlabel('Actual Salary (thousands of dollars)')
    plt.ylabel('Predicted Salary (thousands of dollars)')
    plt.title('Actual vs Predicted Salaries (Test Set)')
    plt.legend()
    st.pyplot(plt)

    # Graph 2: Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='green')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Residual Line')
    plt.xlabel('Predicted Salary (thousands of dollars)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.legend()
    st.pyplot(plt)

    # Explanatory report
    st.subheader("📝 Explanatory Report")
    st.write("""
    ### Interpretation of Results:
    1. **Salary Prediction**:
       - The model predicts the salary based on experience, education level, and technical skills.
       - The prediction is shown in thousands of dollars.

    2. **Model Evaluation**:
       - The **R²** metric indicates how much variability in actual salaries is explained by the model.
       - An R² close to 1 suggests the model is highly accurate.

    3. **Graphs**:
       - **Actual vs Predicted Salaries**: Shows how well the predictions match the actual values.
       - **Residual Plot**: Helps identify patterns in the model's errors.
    """)
else:
    st.error("The CSV file must contain the columns: 'Experience', 'educational_level', 'skills', and 'salary'.")
