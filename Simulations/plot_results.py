import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
plt.style.use('classic')

#Nominal cost dynamic delta 
df3 = pd.read_csv('sarsa_cache1000_request_rate1000_nb_interval40000interval_size1delta10method_Q_learningcoeffcients_[1, 2, 4, 8, 16].csv', header=0)

plt.figure()
df3['Nominal_Cost'].rolling(window=600).mean().plot()
df3['Best_Cost'].rolling(window=600).mean().plot()
df3['Cost_First'].rolling(window=600).mean().plot()
df3['Total_Cost'].rolling(window=600).mean().plot()
plt.legend(['Nominal_Cost', 'Best_Cost','Cost_First','Total_Cost'])
plt.title("Q-Learning with request_rate = 1000, interval size = 1, and delta = 10, 20, 40, 80, 160")
plt.xlabel("Time")

plt.show()