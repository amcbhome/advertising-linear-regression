import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st

# 1. Dataset Initialization
# Data represents 12 months of Advertising Expenses (X) and Net Profit (Y) in £1,000s
data = {
    'Month': np.arange(1, 13),
    'Expenses_X': [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
    'Profit_Y': [15.0, 17.5, 19.0, 22.0, 24.0, 25.0, 28.5, 30.0, 33.0, 35.5, 37.0, 40.0]
}

df = pd.DataFrame(data)

# 2. Reshaping data for Scikit-Learn
X = df['Expenses_X'].values.reshape(-1, 1)
y = df['Profit_Y'].values

# 3. Building the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# 4. Extracting Coefficients
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y)

# Streamlit App
st.set_page_config(page_title="Advertising Profit Calculator", layout="centered")
st.title("Advertising Expense to Net Profit Calculator")
st.write("This app predicts net profit based on advertising expenses using a simple linear regression model.")

st.subheader("Model Details")
st.markdown(f"- **Regression Equation**: Y = {intercept:.2f} + {slope:.2f}X")
st.markdown(f"- **R-squared (R²)**: {r_squared:.4f} ({r_squared*100:.1f}%) -- *This indicates how well the model fits the data.*")

st.markdown("--- ")
st.subheader("Predict Net Profit")

# Input for advertising expense
advertising_expense = st.number_input("Enter Advertising Expense (£1,000s)", min_value=0.0, value=6.5, step=0.5)

if advertising_expense:
    # 5. Predictions
    x_predict = np.array([[advertising_expense]])
    y_predict = model.predict(x_predict)

    # Display results
    st.success(f"Predicted Net Profit for £{advertising_expense:.2f}k Advertising: **£{y_predict[0]:.2f}k**")

st.markdown("--- ")
st.subheader("Data Visualization")

# 7. Visualizing the Data and Regression Line
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, color='blue', label='Actual Data')
ax.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
ax.set_title('Advertising Expenditures vs. Net Profit')
ax.set_xlabel('Advertising Expenses (£1,000s)')
ax.set_ylabel('Net Profit (£1,000s)')
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.caption("To run this Streamlit app locally, save this code as a Python file (e.g., `app.py`) and execute `streamlit run app.py` in your terminal. Ensure you have `streamlit`, `numpy`, `pandas`, `scikit-learn`, and `matplotlib` installed (`pip install streamlit numpy pandas scikit-learn matplotlib`).")
