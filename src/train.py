

from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data # shape (150,4)
y = iris.target # shape (150)
print(iris.feature_names, iris.target_names)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)


model.fit(x_train,y_train)


y_pred = model.predict(x_test)


print("predicitions:",y_pred[:5])
print("true labels:",y_test[:5])


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("accuracy;", accuracy)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


from sklearn.neighbors import KNeighborsClassifier
model12 = KNeighborsClassifier(n_neighbors=5)
model12.fit(x_train, y_train)
y_pred2 = model12.predict(x_test)
print("k-NN accuracy;", accuracy_score(y_test, y_pred2))


model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)

import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

os.makedirs("outputs", exist_ok=True)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
plt.close()

print("Saved: outputs/confusion_matrix.png")

joblib.dump(model12, "outputs/model.joblib")
print("Saved: outputs/model.joblib")
