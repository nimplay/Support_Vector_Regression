# Salary Prediction App using Support Vector Regression (SVR)

![App Screenshot](https://via.placeholder.com/600x400?text=App+Screenshot) <!-- Add a screenshot here if available -->

This is a **Streamlit** web application that predicts salaries based on professional features such as experience, education level, and technical skills. The app uses **Support Vector Regression (SVR)** for predictions and provides visualizations and an explanatory report to help users understand the results.

---

## Features

1. **Salary Prediction**:
   - Predicts salaries based on:
     - **Experience** (years)
     - **Education Level** (Bachelor's, Master's, PhD)
     - **Technical Skills** (1-10 scale).

2. **Manual CSV Upload**:
   - Users can upload their own CSV file with the required columns:
     - `Experience`
     - `educational_level`
     - `skills`
     - `salary`.

3. **Interactive Visualizations**:
   - **Actual vs Predicted Salaries**: Compares the model's predictions with actual salaries.
   - **Residual Plot**: Shows the difference between actual and predicted values.

4. **Explanatory Report**:
   - Provides insights into:
     - How the model works.
     - What the R² score means.
     - How to interpret the graphs.

5. **Custom Styling**:
   - The app features a clean and modern design with custom CSS styles.

---

## Requirements

To run this app, you need the following Python libraries:

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

## Live Version
https://supportvectorregression-mdhjql2mhpdu5oqrueg4qw.streamlit.app/


## You can install these dependencies using:

```bash

 pip install streamlit pandas numpy scikit-learn matplotlib

---

