import pandas as pd
import matplotlib.pyplot as plt

power_plant = pd.read_csv("data/CCPP_data.csv")

x = power_plant['AT']
y = power_plant['PE']

plt.scatter(x, y)
plt.xlabel('Temperature')
plt.ylabel('Energy Output')
plt.show()
