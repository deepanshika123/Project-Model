import pickle

# Load trained model
with open("model/salary_model.pkl", "rb") as file:
    model = pickle.load(file)

experience = float(input("Enter years of experience: "))

prediction = model.predict([[experience]])

print("Predicted Salary:", prediction[0])