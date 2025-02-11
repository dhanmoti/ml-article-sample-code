import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify

# Load dataset
df = pd.read_csv('data/employeesalary.csv')

# Selecting relevant columns (assuming 'experience' and 'salary' exist in dataset)
X = df[['Experience_Years']]  
y = df['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
print(model)

# Create Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    years_experience = float(data['years_experience'])
    prediction = model.predict([[years_experience]])
    return jsonify({'predicted_salary': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
