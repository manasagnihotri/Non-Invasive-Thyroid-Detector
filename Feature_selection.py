import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

x_train = pd.read_csv("https://raw.githubusercontent.com/adityacs103/dataset/main/x_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/adityacs103/dataset/main/y_train.csv")

x_test = pd.read_csv("https://raw.githubusercontent.com/adityacs103/dataset/main/x_test.csv")
y_test = pd.read_csv("https://raw.githubusercontent.com/adityacs103/dataset/main/y_test.csv")

# Applying selectKbest() to reduce the number of features.

def feature_selection(x,y):
   
    obj = SelectKBest(chi2, k=4)
    obj.fit_transform(x,y)
    filter = obj.get_support()
    feature = x.columns
    final_f = feature[filter]
    print(final_f)
    
    return final_f

features = feature_selection(x_train, y_train)
