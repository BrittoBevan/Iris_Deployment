import streamlit as st
import pickle
import numpy as np
#import sklearn

#Load saved model
with open("model.pkl","rb")as f:
    model=pickle.load(f)

#Steramlit UI
st.title("Iris Flower Classifier")
st.write("Enter the features below: ")

#Input fields
sl=st.number_input("Sepal Length",min_value=4.0,max_value=8.0)
sw=st.number_input("Sepal Width",min_value=2.0,max_value=5.0)
pl=st.number_input("Petal Length",min_value=1.0,max_value=7.0)
pw=st.number_input("Petal Width",min_value=0.1,max_value=3.0)

#Predict button
if st.button("Predict"):
    prediction=model.predict([[sl,sw,pl,pw]])
    classes=["Setosa","Versicolor","Virginica"]
    st.write(f"Prediction: {classes[prediction[0]]}")
