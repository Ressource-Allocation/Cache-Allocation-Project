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

#Plot Q-learning results with request_rate = 10 and delta = 20
df = pd.read_csv('sarsa_cache1000_request_rate10_nb_interval40000interval_size1delta20_q_learning.csv', header=0)

plt.figure()
df['Nominal_Cost'].rolling(window=600).mean().plot()
df['Best_Cost'].rolling(window=600).mean().plot()
df['Cost_First'].rolling(window=600).mean().plot()
df['Total_Cost'].rolling(window=600).mean().plot()
plt.legend(['Nominal_Cost', 'Best_Cost','Cost_First','Total_Cost'])
plt.title("Q-Learning")
plt.xlabel("Time")

plt.show()