
import mlflow
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as pyplot
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import metrics as mt
import subprocess
import pickle
import altair as alt
from codecarbon import EmissionsTracker

st.set_page_config(page_title="Heart Attack Analysis App")

tracker = EmissionsTracker()
tracker.start()
tracker.stop()


# Read the dataset
df = pd.read_csv("heartStats.csv")
df = df.rename(columns={'sex': 'Sex','age': 'Age','cp': 'Chest Pain','trtbps': 'Resting Blood Pressure','chol': 'Cholesterol','fbs': 'Fasting Blood Sugar','restecg': 'Resting ECG','thalachh': 'Maximum Heart Rate','exng': 'Exercise Induced Angina','oldpeak': 'Exercise-induced ST Depression','slp': 'Peak Exercise ST Segment','caa': '# of Major Vessels Covered By Fluoroscopy','thall': 'Thalassemia Reversable Defect','output': 'Heart Attack Prediction'})

# Sidebar for navigation
app_mode = st.sidebar.selectbox('Select page',['Introduction','Visualization','Prediction','Deployment','Analysis'])

if app_mode == 'Introduction':
  # Set the title of the web app
  st.title("Heart Attack Prediction App")
  #gif
  gif_path = 'HeartAttackImage.gif'
  width=250
  st.image(gif_path, width=width)
  
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
  st.image(image_heart, width=250)

if app_mode == "Visualization":
  st.markdown("# :red[Visualization]")
  # Visualization page for plotting graphs
  list_variables = df.columns
  symbols = st.multiselect("Select two variables",list_variables, ["Age", "Chest Pain"])
  width1 = st.sidebar.slider("plot width", 1, 25, 10)
  tab1, tab2, tab3, tab4 = st.tabs(["Line Chart", "Bar Chart", "Correlation", "Pair Plot"])

  if tab1.button("Show Line Chart"):
      st.line_chart(data=df, x=symbols[0], y=symbols[1], width=0, height=0, use_container_width=True)
      tab1.subheader("Line Chart")
      #tab1.line_chart(data=df, x=symbols[0],y=symbols[1], width=0, height=0, use_container_width=True)
      tab1.write(" ")

  if tab2.button("Show Bar Chart"):
      st.bar_chart(data=df, x=symbols[0], y=symbols[1], use_container_width=True)
      tab2.subheader("Bar Chart Tab")
    
  if tab3.button("Show Correlation Grid"):
      fig, ax = plt.subplots(figsize=(width1, width1))
      sns.heatmap(df.corr(), cmap=sns.cubehelix_palette(8), annot=True, ax=ax)
      tab3.write(fig)

  if tab4.button("Show Pairplot"):
      st.markdown("### Pairplot")
      df2 = df
      fig3 = sns.pairplot(df2)
      st.pyplot(fig3)
  
  #image
  image_heart = Image.open('heartclipart2.png')
  st.image(image_heart, width=250)

if app_mode == "Prediction":
  # Prediction page to predict wine quality

  
  model_mode = st.sidebar.selectbox("Select Model",["KNN", "DecisionTreeClassifier", "LogisticRegression"])
  
  if model_mode == 'KNN':
    st.markdown("# :red[KNN Prediction]")
    X = df.drop(labels="Heart Attack Prediction", axis=1)
    y = df["Heart Attack Prediction"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.7)
    lm = KNeighborsClassifier()
    lm.fit(X_train,y_train)
    predictions = lm.predict(X_test)
    st.write("Heart Attack Prediction", predictions)

  
    # Display performance metrics of the model
    accuracy = accuracy_score(y_test, predictions)
    accuracy2 = int(round(accuracy, 2) * 100)
    st.write("1 The accuracy of this model is", accuracy2, "%")


  if model_mode == 'DecisionTreeClassifier':
    st.markdown("# :red[DecisionTreeClassifier Prediction]")
    X = df.drop(labels="Heart Attack Prediction", axis=1)
    y = df["Heart Attack Prediction"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.7)
    lm = DecisionTreeClassifier()
    lm.fit(X_train,y_train)
    predictions = lm.predict(X_test)
    st.write("Heart Attack Prediction", predictions)
    

    # Display performance metrics of the model
    accuracy = accuracy_score(y_test, predictions)
    accuracy2 = int(round(accuracy, 2) * 100)
    st.write("1 The accuracy of this model is", accuracy2, "%")


  if model_mode == 'LogisticRegression':
    st.markdown("# :red[LogisticRegression Prediction]")
    X = df.drop(labels="Heart Attack Prediction", axis=1)
    y = df["Heart Attack Prediction"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.7)
    lm = LogisticRegression()
    lm.fit(X_train,y_train)
    predictions = lm.predict(X_test)
    st.write("Heart Attack Prediction", predictions)
   
    
    # Display performance metrics of the model
    accuracy = accuracy_score(y_test, predictions)
    accuracy2 = int(round(accuracy, 2) * 100)
    st.write("1 The accuracy of this model is", accuracy2, "%")

  #image
  image_heart = Image.open('heartclipart2.png')
  st.image(image_heart, width=250)

if app_mode == 'Deployment':
    # Deployment page for model deployment
    st.markdown("# :red[Deployment]")
    #id = st.text_input('ID Model', '/content/mlruns/1/0ad40de668d6475dab9dccad85438f40/artifacts/top_model_v1')

    # Load model for prediction
    #logged_model = f'./mlruns/1/a768fe9670c94e098f3ab45564f0db8d/artifacts/top_model_v1'
    #loaded_model = mlflow.pyfunc.load_model(logged_model)
    model_mode = st.sidebar.selectbox("Select Model",["LinearRegression","DecisionTreeClassifier", "LogisticRegression"])
    if model_mode == 'KNN': 
      model_filename ='kneighbors.pkl'
      with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)

    if model_mode == 'DecisionTreeClassifier': 
      model_filename ='DecisionTreeClassifierName.pkl'
      with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)

    if model_mode == 'LogisticRegression': 
      model_filename ='LogisticRegressionName.pkl'
      with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)

    deploy_df= df.drop(labels='Heart Attack Prediction', axis=1)
    list_var = deploy_df.columns
    #st.write(target_choice)

    number1 = st.number_input("Age", value=50)
    number2 = st.selectbox("Sex (Female = 0 /// Male = 1)", [0, 1])
    number3 = st.selectbox("Chest Pain (Typical Angina = 0 /// Atypical Angina = 1 /// Non-anginal Pain = 2 /// Asymptomatic = 3", [0, 1, 2, 3])
    number4 = st.number_input("Systolic Blood Pressure (Normal < 120   ///   Hypertension > 140)", value=130)
    number5 = st.number_input("Cholesterol (Normal < 200   ///   High > 240", value=250)
    number6 = st.selectbox("Resting Blood Sugar (1 = RBS > 120mg   ///   0 = RBS < 120)", [0, 1])
    number7 = st.selectbox("Resting ECG (0 = Normal /// 1 = ST-T wave normality /// 2 = Left ventricular hypertrophy", [0, 1, 2])
    number8 = st.number_input("Maximum Heart Rate (Average Max Heart Rate = (220 - [age of patient])", value=170)
    number9 = st.selectbox("Exercise Induced Angina (Yes = 1 /// No = 0", [0, 1])
    number10 = st.selectbox("Previous Peak", [0, 1, 2, 3, 4, 5, 6, 7])
    number11 = st.selectbox("Slope", [0, 1, 2])
    number12 = st.selectbox("Number of Major Vessels Covered By Fluoroscopy", [0, 1, 2, 3])
    number13 = st.selectbox("Thallium Reversable Defect", [0, 1, 2, 3])

    data_new = pd.DataFrame({deploy_df.columns[0]:[number1], deploy_df.columns[1]:[number2], deploy_df.columns[2]:[number3],
         deploy_df.columns[3]:[number4], deploy_df.columns[4]:[number5], deploy_df.columns[5]:[number6], deploy_df.columns[6]:[number7],
         deploy_df.columns[7]:[number8], deploy_df.columns[8]:[number9],deploy_df.columns[9]:[number10],deploy_df.columns[10]:[number11],deploy_df.columns[11]:[12],deploy_df.columns[12]:[13]})
    # Predict on a Pandas DataFrame.
    #import pandas as pd
    st.write("Prediction :", np.round(loaded_model.predict(data_new)[0],2))

    #image
    image_heart = Image.open('heartclipart2.png')
    st.image(image_heart, width=250)

if app_mode == "Analysis":
  st.markdown("# :red[Analysis]")
  list_variables = df.columns


  # Allow users to select two variables from the dataset for visualization
  symbols = st.multiselect("Select two variables", list_variables, ["Age", "Heart Attack Prediction"])

  # Create a slider in the sidebar for users to adjust the plot width
  width1 = st.sidebar.slider("plot width", 1, 25, 10)

  # Display a bar chart for the selected variables
  st.bar_chart(data=df, x=symbols[0], y=symbols[1], use_container_width=True)


  st.markdown("1. Risk of heart attack is most highly correlated with variables Max Heart Rate, Peak Exercise ST Segment, and Chest Pain.")
  st.markdown("1. Risk of heart attack is highest between 41 and 59 years of age, according to this data set.")


if app_mode == "CLICK":
  image_joke = Image.open('thatsallfolks.jpg')
  st.image(image_joke, width=750)
