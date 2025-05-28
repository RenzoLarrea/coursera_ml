import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

power_plant = pd.read_csv("data/CCPP_data.csv")

log_x = np.log(power_plant['V'].values).reshape(-1, 1)
y = power_plant['PE']

plt.scatter(log_x, y)
plt.xlabel('Exhaust Vacuum (log)')
plt.ylabel('Energy Output')
plt.show()
