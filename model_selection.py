# Importing the necessary libraries.
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers
# import logging


x_train =  pd.read_csv("https://raw.githubusercontent.com/adityacs103/dataset/main/x_train.csv")
y_train =  pd.read_csv("https://raw.githubusercontent.com/adityacs103/dataset/main/y_train.csv")

x_test =  pd.read_csv("https://raw.githubusercontent.com/adityacs103/dataset/main/x_test.csv")
y_test =  pd.read_csv("https://raw.githubusercontent.com/adityacs103/dataset/main/y_test.csv")

x_valid =  pd.read_csv("https://raw.githubusercontent.com/adityacs103/dataset/main/x_valid.csv")
y_valid =  pd.read_csv("https://raw.githubusercontent.com/adityacs103/dataset/main/y_valid.csv")

# Creating pipelies.
pipe1 = Pipeline([("minmax_scalar", MinMaxScaler()), ("logistic_regression", LogisticRegression())])

pipe2 = Pipeline([("minmax_scalar", MinMaxScaler()), ("KNN", KNeighborsClassifier())])

pipe3 = Pipeline([("minmax_scalar", MinMaxScaler()), ("svm", SVC())])

pipe4 = Pipeline([("minmax_scalar", MinMaxScaler()), ("XGboost", XGBClassifier())])

pipe5 = Pipeline([("minmax_scalar", MinMaxScaler()), ("decision_tree", DecisionTreeClassifier())])

pipe6 = Pipeline([("minmax_scalar", MinMaxScaler()), ("random_forest", RandomForestClassifier())])

def build_ann():
    
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[54]),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')])
    
    model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"])
    
    return model

ann_model = build_ann()

# Fitting the pipelines
pipelines = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6]

for pipe in pipelines:
    pipe.fit(x_train, y_train)
    
callback = keras.callbacks.EarlyStopping(monitor = "val_binary_accuracy", patience=3, restore_best_weights=True)

history = ann_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=100, 
    callbacks = [callback])

# Predicting
pred1 = pipe1.predict(x_valid)
pred2 = pipe2.predict(x_valid)
pred3 = pipe3.predict(x_valid)
pred4 = pipe4.predict(x_valid)
pred5 = pipe5.predict(x_valid)
pred6 = pipe6.predict(x_valid)

# Comparing the result of each pipeline and selecting the best pipeline. 

