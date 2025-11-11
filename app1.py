# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.tree import plot_tree

# --- Load trained model ---
@st.cache_resource
def load_model():
    with open('iris_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

dtree = load_model()

# --- Load dataset for display ---
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data = pd.read_csv(url, names=column_names)
    return data

data = load_data()

# --- Streamlit UI ---
st.title("🌸 Iris Flower Classifier (Decision Tree Model)")
st.markdown("This app uses a **pre-trained Decision Tree model** to classify Iris flower species.")

# --- Dataset Preview ---
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# --- Prediction Section ---
st.subheader("🔮 Make a Prediction")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length", float(data['sepal_length'].min()), float(data['sepal_length'].max()), 5.0)
    sepal_width = st.slider("Sepal Width", float(data['sepal_width'].min()), float(data['sepal_width'].max()), 3.5)
with col2:
    petal_length = st.slider("Petal Length", float(data['petal_length'].min()), float(data['petal_length'].max()), 1.4)
    petal_width = st.slider("Petal Width", float(data['petal_width'].min()), float(data['petal_width'].max()), 0.2)

# Prepare input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict
if st.button("Predict Flower Type"):
    prediction = dtree.predict(input_data)[0]
    st.success(f"🌼 The model predicts this flower is: **{prediction}**")

