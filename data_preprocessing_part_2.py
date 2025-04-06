# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler 



df =  pd.read_csv("https://raw.githubusercontent.com/adityacs103/dataset/main/data_processed_1.csv")

# We saw that the null values were imputed by a "?". To keep things simple I re-converted the "?" into numpy null value.
df.replace({"?":np.nan}, inplace=True)

# The columns "TBG" and "T3" has a lot of null values. Imputing these null values might reduce variance and might tease 
# the model into giving more importance to a particular case.
df.drop(columns=["TBG", "T3"], inplace=True)
 
# than male "M" or "female" "F".
df.sex.fillna("unknown", inplace=True)

# Coverting the datatype of continous features into numeric type.
df.TSH = pd.to_numeric(df.TSH)
df.TT4 = pd.to_numeric(df.TT4)
df.T4U = pd.to_numeric(df.T4U)
df.FTI = pd.to_numeric(df.FTI)

# Removing outliers 
index_age = df[df["age"]>100].index
df.drop(index_age, inplace=True)

# removing TSH value higher than 15. That's very rare.
index_tsh = df[df["TSH"]>15].index
df.drop(index_tsh, inplace=True)

# Encoding the categorical features. 
df_dummy = pd.get_dummies(df)


# Imputing null values using KNNImputer.
def Imputation(df):
    imputer = KNNImputer(n_neighbors=3)
    df_1 = imputer.fit_transform(df)
    df_2 = pd.DataFrame(df_1, columns=df.columns)
    return df_2
    

df_final = Imputation(df_dummy[:7000])
# Splitting the data into train, test and validation to prevent data leakage.
validation_data = df_dummy[7000:]
x_train, x_test, y_train, y_test = train_test_split(df_final.drop(columns="outcome"), df_final["outcome"], test_size=0.2)

valid_final = Imputation(validation_data)

# Fixing the imbalanced data by creating duplicate records.
def balance_data(x,y):    
    ros = RandomOverSampler(random_state=42)
    x_sample, y_sample = ros.fit_resample(x, y)
    return x_sample, y_sample

x_train, y_train = balance_data(x_train, y_train)
x_test, y_test = balance_data(x_test, y_test)

x_valid, y_valid = balance_data(valid_final.drop(columns="outcome"), valid_final["outcome"])

x_train.to_csv("x_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)

x_test.to_csv("x_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

x_valid.to_csv("x_valid.csv", index=False)
y_valid.to_csv("y_valid.csv", index=False)
