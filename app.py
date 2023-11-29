
import mlflow
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as pyplot
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import metrics as mt
import subprocess
import pickle

# Set the title of the web app
st.title("Heart Attack Analysis")

# Read the dataset
df = pd.read_csv("heartStats.csv")
df = df.rename(columns={'sex': 'Sex','age': 'Age','cp': 'Chest Pain','trtbps': 'Resting Blood Pressure','chol': 'Cholesterol','fbs': 'Fasting Blood Sugar','restecg': 'Resting ECG','thalachh': 'Maximum Heart Rate','exng': 'Exercise Induced Angina','oldpeak': 'Exercise-induced ST Depression','slp': 'Peak Exercise ST Segment','caa': '# of Major Vessels Covered By Fluoroscopy','thall': 'Thalassemia Reversable Defect','output': 'Heart Attack Prediction'})

gif_path = 'HeartAttackImage.gif'
width=250
st.image(gif_path, width=width)

# Sidebar for navigation
app_mode = st.sidebar.selectbox('Select page',['Introduction','Visualization','Prediction','Deployment'])

if app_mode == 'Introduction':
  # Introduction page allowing user to view dataset rows
  num = st.number_input('No of Rows',5,10)
  st.dataframe(df.head(num))

  # Display statistical description of the dataset
  st.dataframe(df.describe())

  # Calculate and display the percentage of missing values in the dataset
  dfnull = df.isnull().sum()/len(df)*100
  totalmiss = dfnull.sum().round(2)
  st.write("Percentage of missing value in my dataset",totalmiss)

  #image
  image_heart = Image.open('heartclipart2.png')
  st.image(image_heart, width=100)

if app_mode == "Visualization":
  # Visualization page for plotting graphs
  list_variables = df.columns
  symbols = st.multiselect("Select two variables",list_variables, ["age", "cp"])
  st.line_chart(data=df, x=symbols[0], y=symbols[1])
  st.bar_chart(data=df, x=symbols[0], y=symbols[1])

  # Pairplot for selected variables
  df2 = df[[list_variables[0],list_variables[1],list_variables[2],list_variables[3]]]
  fig = sns.pairplot(df2)
  st.pyplot(fig)

  #image
  image_heart = Image.open('heartclipart2.png')
  st.image(image_heart, width=100)

if app_mode == "Prediction":
  # Prediction page to predict wine quality
  st.write("Heart Attack Prediction")
  X = df.drop(labels="output", axis=1)
  y = df["output"]
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.7)
  lm = LinearRegression()
  lm.fit(X_train,y_train)
  predictions = lm.predict(X_test)
  st.write(predictions)

  # Display performance metrics of the model
  variance = np.round(metrics.explained_variance_score(y_test, predictions)*100,2)
  st.write("1 The models explains",variance )
  mae = np.round(metrics.mean_absolute_error(y_test,predictions),2)
  st.write("2 The mean absolute error", mae)

  # Calculating additional metrics
  mae = np.round(mt.mean_absolute_error(y_test, predictions ),2)
  mse = np.round(mt.mean_squared_error(y_test, predictions),2)
  r2 = np.round(mt.r2_score(y_test, predictions),2)

  #image
  image_heart = Image.open('heartclipart2.png')
  st.image(image_heart, width=100)

if app_mode == 'Deployment':
    # Deployment page for model deployment
    st.markdown("# :violet[Deployment 🚀]")
    #id = st.text_input('ID Model', '/content/mlruns/1/0ad40de668d6475dab9dccad85438f40/artifacts/top_model_v1')

    # Load model for prediction
    #logged_model = f'./mlruns/1/a768fe9670c94e098f3ab45564f0db8d/artifacts/top_model_v1'
    #loaded_model = mlflow.pyfunc.load_model(logged_model)
    model_filename ='model.pkl'
    with open(model_filename, 'rb') as file:
      loaded_model = pickle.load(file)


    df = pd.read_csv("heartStats.csv")
    df = df.rename(columns={'sex': 'Sex','age': 'Age','cp': 'Chest Pain','trtbps': 'Resting Blood Pressure','chol': 'Cholesterol','fbs': 'Fasting Blood Sugar','restecg': 'Resting ECG','thalachh': 'Maximum Heart Rate','exng': 'Exercise Induced Angina','oldpeak': 'Exercise-induced ST Depression','slp': 'Peak Exercise ST Segment','caa': '# of Major Vessels Covered By Fluoroscopy','thall': 'Thalassemia Reversable Defect','output': 'Heart Attack Prediction'})
    deploy_df= df.drop(labels='output', axis=1)
    list_var = deploy_df.columns
    #st.write(target_choice)

    number1 = st.number_input(deploy_df.columns[0],0.7)
    number2 = st.number_input(deploy_df.columns[1],0.04)
    number3 = st.number_input(deploy_df.columns[2],1.1)
    number4 = st.number_input(deploy_df.columns[3],0.05)
    number5 = st.number_input(deploy_df.columns[4],25)
    number6 = st.number_input(deploy_df.columns[5],20)
    number7 = st.number_input(deploy_df.columns[6],0.98)
    number8 = st.number_input(deploy_df.columns[7],1.9)
    number9 = st.number_input(deploy_df.columns[8],0.4)
    number10 = st.number_input(deploy_df.columns[9],9.4)
    number11 = st.number_input(deploy_df.columns[10],5)
    number12 = st.number_input(deploy_df.columns[11],2)
    number13 = st.number_input(deploy_df.columns[10],2)

    data_new = pd.DataFrame({deploy_df.columns[0]:[number1], deploy_df.columns[1]:[number2], deploy_df.columns[2]:[number3],
         deploy_df.columns[3]:[number4], deploy_df.columns[4]:[number5], deploy_df.columns[5]:[number6], deploy_df.columns[6]:[number7],
         deploy_df.columns[7]:[number8], deploy_df.columns[8]:[number9],deploy_df.columns[9]:[number10],deploy_df.columns[10]:[number11],deploy_df.columns[11]:[12],deploy_df.columns[12]:[13]})
    # Predict on a Pandas DataFrame.
    #import pandas as pd
    st.write("Prediction :", np.round(loaded_model.predict(data_new)[0],2))

    #image
    image_heart = Image.open('heartclipart2.png')
    st.image(image_heart, width=100)
