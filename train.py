import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


data = pd.read_csv("data/Salary.csv")


X = data[["YearsExperience"]]
y = data["Salary"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)



plt.scatter(X, y, label="Actual Data")


plt.plot(X, model.predict(X), label="Regression Line")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Years of Experience")
plt.savefig("salary_regression.png")
plt.show()


print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)


with open("model/salary_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")