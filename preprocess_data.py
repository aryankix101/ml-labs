import pandas as pd
from sklearn.model_selection  import train_test_split


df = pd.read_csv (r'C:\Users\aryan\OneDrive\Desktop\projects\machine-learning-tjhsst\april2012dataset.csv')

for name, values in df.iteritems():
    missing_values_count = 0
    total_values = len(values)
    for value in values:
        if value==" ":
            missing_values_count+=1
    if missing_values_count/total_values>0.5:
        df = df.drop(name, 1)
    else:
        print(df[name].mode()[0])
        df[name] = df[name].replace(" ", df[name].mode()[0])

train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

train.to_csv("train.csv")
test.to_csv("test.csv")
df.to_csv('April_2012_Mobile_preprocessed.csv')