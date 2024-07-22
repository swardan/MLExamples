
# sklearn - ML Library
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# (1) Dataset Analysis
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df["target"] = cancer.target


X = cancer.data # features
y = cancer.target # targets



# (2) Selecting ML Model - KNN Classifier




# (3) Training Model
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)


## Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)  # parameters!!
knn.fit(X_train, y_train) # trains KNN using samples and targets



# (4) Evaluating Results 
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy:",accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# (5) Optimizing Hyperparameters
"""
    KNN : Hyperparameter = K
        K : 1,2,3,...N
        Accuracy : %A, %B, %C...
"""
k_values = []
accuracy_values = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    k_values.append(k)
    accuracy_values.append(accuracy)

plt.figure()
plt.plot(k_values, accuracy_values, linestyle="-", marker = "o")
plt.grid()
plt.xlabel("K Values")
plt.ylabel("Accuracies")
plt.xticks(k_values)



