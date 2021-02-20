import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
df=pd.read_csv("sheet1.csv")
X=df.drop(columns=['DifficultyLevel'])
y=df['DifficultyLevel']
model=DecisionTreeClassifier()
prediction=model.fit(X,y)

pickle.dump(prediction,open('levl.pki','wb'))
