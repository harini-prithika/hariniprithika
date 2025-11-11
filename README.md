# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Load the Iris dataset
iris = load_iris()
X = iris.data          # features: sepal & petal measurements
y = iris.target        # labels: 0=setosa, 1=versicolor, 2=virginica

# 2️⃣ Split into training and testing data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Feature scaling (optional but good practice)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4️⃣ Train a classification model (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Predict on test data
y_pred = model.predict(X_test)

# 6️⃣ Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Detailed performance report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
