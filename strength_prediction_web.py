loaded_model = pickle.load(open("trained_model.sav", 'rb'))
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the saved model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))


# Function for prediction
def strength_prediction(input_data):
    # Reshape the input data as we are predicting for one instance
    input_data_reshaped = input_data.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

# Function for scatter plot
def scatter_plot(df, x_column, y_column):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_column], df[y_column], color='skyblue')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title('Scatter Plot')
    st.pyplot(plt)

# Function for histogram
def histogram(df, column):
    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins='auto', color='skyblue')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column}')
    st.pyplot(plt)

# Sidebar for navigation
with st.sidebar:
    st.markdown("## Navigation")

    # Icon definitions
    strength_icon = "📈"
    visualization_icon = "📊"

    # Navigation options with icons
    selected = st.radio(
        "Go to",
        (
            f"{strength_icon} Strength Prediction",
            f"{visualization_icon} Data Visualization"
        )
    )

# Strength Prediction Page
if selected.startswith(strength_icon):
    # Page title
    st.title('Cement Strength Prediction')

    # Read the dataset
    df = pd.read_csv(r"C:\Users\User\ML Project cement\concrete.csv")

    # Select the first row as sample input
    sample_input = df.drop('strength', axis=1).iloc[0].values

    # Create input fields for sample input
    input_values = []
    for i, column in enumerate(df.columns[:-1]):
        value = st.number_input(column, value=sample_input[i])
        input_values.append(value)

    # Convert input values to numpy array
    sample_input = np.array(input_values)

    # Predict button
    if st.button('Predict'):
        prediction = strength_prediction(sample_input)
        st.success(f'Predicted Strength: {prediction[0]}')

# Data Visualization Page
elif selected.startswith(visualization_icon):
    # Read the dataset
    df = pd.read_csv(r"C:\Users\User\ML Project cement\concrete.csv")

    # Page title
    st.title('Data Visualization')

    # Scatter Plot
    st.header('Scatter Plot')
    x_column = st.selectbox('Select X-axis column', df.columns)
    y_column = st.selectbox('Select Y-axis column', df.columns)
    if st.button('Plot Scatter'):
        scatter_plot(df, x_column, y_column)

    # Histogram
    st.header('Histogram')
    column = st.selectbox('Select column', df.columns)
    if st.button('Plot Histogram'):
        histogram(df, column)
