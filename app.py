# importing the necessary libraries
import pickle as pkl
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# load the trained model from the pickle file
@st.cache_resource   # decorator to cache the loaded model
def load_model():    # creating a function by the name load_model
    with open('iris_model.pkl', 'rb') as file:    # opening the pickle  file in read binary mode
        model = pickle.load(file)  # loading the model from the file
    return model       # returning the model

dtree = load_model # callin the function to load the model

# --- load datasetfor display ---
@st.cache_data   # decorator to cache the loaded data
def load_data(): # creatng a function bythe name load_data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"  # url of the iris data
    column_names = ['sepal_length','sepal_width','petal_length','petal_width', 'class']  # defining the columns
    data = pd.read_csv(url, names = column_names)   # reading the dataset from the url
    return data # returning the data

data = load_data()   # calling the function to load data

# --- Streamlit UI ---
st.title('🌸 Iris Flower Classifier (Decision Tree Model)')
st.markdown('This app uses a **pre-trained Decision Tree model** to classify Iris Flower Species.')

# --- Dataset view ---
st.subheader('📊 Dataset Preview')
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
