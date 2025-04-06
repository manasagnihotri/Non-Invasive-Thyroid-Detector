# Importing the necessary libraries
import pandas as pd

df =  pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/thyroid0387.data")


# Saving the first character in a new column. Because thats what matter for this problem statement.

df["outcome"] = df["-[840801013]"].str[0]
df.drop(columns="-[840801013]", inplace=True)


# Replacing all possible disease outcomes into one category - "yes".
list = ['S', 'F', 'A', 'R', 'I', 'M', 'N', 'G', 'K', 'L', 'Q', 'J',
       'C', 'O', 'H', 'D', 'P', 'B', 'E']
df['outcome'].replace(to_replace=list, value="yes", inplace=True)


# Replacing the binary outputs into integer values 0 and 1 for simplicity.
df.outcome.replace({"-":0, "yes":1}, inplace=True)


# Here I replace the column names with more simple and understandable form.
df.rename(columns = {"29":"age", "F":"sex", "f":"thyroxine", "f.1":"query_thyroxine", "f.2":"medication","f.3":"sick", 
                        "f.4":"pregnant", "f.5":"surgery", "f.6":"I131_treatment", "t":"query_hypothyroid", 
                        "f.7":"query_hyperthyroid", "f.8":"lithium", "f.9":"goitre", "f.10":"tumor", "f.11":"hypopituitary", 
                        "f.12":"psych", "t.1":"TSH_measured","0.3":"TSH", "f.13":"T3_measured", "?":"T3", 
                        "f.14":"TT4_measured", "?.1":"TT4", "f.15":"T4U_measured", "?.2":"T4U", "f.16":"FTI_measured", 
                        "?.3":"FTI", "f.17":"TBG_measured", "?.4":"TBG", "other":"referral_source"}, inplace=True)



df.to_csv("data_processed_1.csv", index=False)
