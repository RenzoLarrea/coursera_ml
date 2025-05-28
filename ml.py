import pandas as pd
import numpy as np
from sklearn import linear_model, tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
import matplotlib.pyplot as plt

mape_scorer = make_scorer(mean_absolute_percentage_error)
power_plant = pd.read_csv("data/CCPP_data.csv")

power_plant['log_V'] = np.log(power_plant['V'])
X = power_plant[['log_V', 'AT', 'AP', 'RH']]
decisiontreeX = power_plant[['V', 'AT', 'AP', 'RH']]
y = power_plant['PE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
decisiontreeX_train, decisiontreeX_test = train_test_split(decisiontreeX, test_size=0.2, random_state=30)

model1 = linear_model.LinearRegression()
model1_score = cross_val_score(model1, X_train, y_train, cv=5, scoring=mape_scorer).mean()

model2 = linear_model.Lasso(alpha=.04, random_state=30)
model2_score = cross_val_score(model2, X_train, y_train, cv=5, scoring=mape_scorer).mean()

model3 = tree.DecisionTreeRegressor(max_depth=11, random_state=30)
model3_score = cross_val_score(model3, decisiontreeX_train, y_train, cv=5, scoring=mape_scorer).mean()

print(f"Linear Reg MAPE (validation): {model1_score:.3%}")
print(f"Lasso Reg MAPE (validation): {model2_score:.3%}")
print(f"Decision Tree Reg MAPE (validation): {model3_score:.2%}")
print("")
print("As the decision tree reg has the lowest MAPE validation score, we choose that as our model")
print("")

model3.fit(decisiontreeX_train, y_train)
y_predictions = model3.predict(decisiontreeX_test)
mape = mean_absolute_percentage_error(y_test, y_predictions)

print(f"Final MAPE from test: {mape:.2%}")
x = y
plt.scatter(y_predictions, y_test)
plt.plot(x, y, color='red')
plt.xlabel('Predicted Energy Output')
plt.ylabel('Actual Energy Output')
plt.grid(True)
plt.show()
