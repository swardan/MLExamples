# Modelling Iris Dataset with Decision Tree



# (1) DATASET UPLOADING AND ANALYSIS
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target


# (2) SELECTING TEST-TRAIN SIZE FOR MODEL
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0) 



# (3) MODEL UPLOADING - DECISION TREE
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# (4) MODEL TRAINING
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

# (5) EVALUATING RESULTS
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_predict)
print(conf_matrix)

# to see in visual
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

index = ['setosa','versicolor','virginica']
columns = ['setosa','versicolor','virginica'] 

confusions = pd.DataFrame(conf_matrix,index,columns)
plt.figure(figsize=(10,6))
sns.heatmap(confusions,annot=True)
