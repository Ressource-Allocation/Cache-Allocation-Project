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

#Plot SARSA results with request_rate = 10 and delta = 20
df1 = pd.read_csv('sarsa_cache1000_request_rate10_nb_interval40000interval_size1delta20_SARSA.csv', header=0)
plt.figure()
df1['Nominal_Cost'].rolling(window=600).mean().plot()
df1['Best_Cost'].rolling(window=600).mean().plot()
df1['Cost_First'].rolling(window=600).mean().plot()
df1['Total_Cost'].rolling(window=600).mean().plot()
plt.legend(['Nominal_Cost', 'Best_Cost','Cost_First','Total_Cost'])
plt.title("SARSA")
plt.xlabel("Time")

plt.show()

#Plot SARSA results with request_rate = 1000 and delta = 20
df2 = pd.read_csv('sarsa_cache1000_request_rate1000_nb_interval40000interval_size1delta20_SARSA.csv', header=0)
plt.figure()
df2['Nominal_Cost'].rolling(window=600).mean().plot()
df2['Best_Cost'].rolling(window=600).mean().plot()
df2['Cost_First'].rolling(window=600).mean().plot()
df2['Total_Cost'].rolling(window=600).mean().plot()
plt.legend(['Nominal_Cost', 'Best_Cost','Cost_First','Total_Cost'])
plt.title("SARSA with request_rate = 1000 and delta = 20")
plt.xlabel("Time")

plt.show()

#Plot Q-learning results with request_rate = 1000 and delta = 20
df3 = pd.read_csv('sarsa_cache1000_request_rate1000_nb_interval40000interval_size1delta20_q_learning.csv', header=0)
plt.figure()
df3['Nominal_Cost'].rolling(window=600).mean().plot()
df3['Best_Cost'].rolling(window=600).mean().plot()
df3['Cost_First'].rolling(window=600).mean().plot()
df3['Total_Cost'].rolling(window=600).mean().plot()
plt.legend(['Nominal_Cost', 'Best_Cost','Cost_First','Total_Cost'])
plt.title("Q-learning with request_rate = 1000 and delta = 20")
plt.xlabel("Time")

plt.show()

#Plot Q-learning results with request_rate = 1000 and delta = 100
df4 = pd.read_csv('sarsa_cache1000_request_rate1000_nb_interval40000interval_size1delta100_q_learning.csv', header=0)
plt.figure()
df4['Nominal_Cost'].rolling(window=600).mean().plot()
df4['Best_Cost'].rolling(window=600).mean().plot()
df4['Cost_First'].rolling(window=600).mean().plot()
df4['Total_Cost'].rolling(window=600).mean().plot()
plt.legend(['Nominal_Cost', 'Best_Cost','Cost_First','Total_Cost'])
plt.title("Q-learning with request_rate = 1000 and delta = 100")
plt.xlabel("Time")

plt.show()

#Plot SARSA results with request_rate = 1000 and delta = 100
df5 = pd.read_csv('sarsa_cache1000_request_rate1000_nb_interval40000interval_size1delta100_SARSA.csv', header=0)
plt.figure()
df5['Nominal_Cost'].rolling(window=600).mean().plot()
df5['Best_Cost'].rolling(window=600).mean().plot()
df5['Cost_First'].rolling(window=600).mean().plot()
df5['Total_Cost'].rolling(window=600).mean().plot()
plt.legend(['Nominal_Cost', 'Best_Cost','Cost_First','Total_Cost'])
plt.title("SARSA with request_rate = 1000 and delta = 100")
plt.xlabel("Time")

plt.show()

#Plot SARSA results with request_rate = 1000 and delta = 100 (total_cost) 
df6 = pd.read_csv('sarsa_cache1000_request_rate1000_nb_interval40000interval_size1delta100_SARSA_total.csv', header=0)
plt.figure()
df6['Nominal_Cost'].rolling(window=600).mean().plot()
df6['Best_Cost'].rolling(window=600).mean().plot()
df6['Cost_First'].rolling(window=600).mean().plot()
df6['Total_Cost'].rolling(window=600).mean().plot()
plt.legend(['Nominal_Cost', 'Best_Cost','Cost_First','Total_Cost'])
plt.title("SARSA with request_rate = 1000 and delta = 100 (total_cost)")
plt.xlabel("Time")

plt.show()

#Plot Q-learning results with request_rate = 1000 and delta = 100 (total_cost) 
df7 = pd.read_csv('sarsa_cache1000_request_rate1000_nb_interval40000interval_size1delta100_q_learning_total.csv', header=0)
plt.figure()
df7['Nominal_Cost'].rolling(window=600).mean().plot()
df7['Best_Cost'].rolling(window=600).mean().plot()
df7['Cost_First'].rolling(window=600).mean().plot()
df7['Total_Cost'].rolling(window=600).mean().plot()
plt.legend(['Nominal_Cost', 'Best_Cost','Cost_First','Total_Cost'])
plt.title("Q-learning with request_rate = 1000 and delta = 100 (total_cost)")
plt.xlabel("Time")

plt.show()

#Plot Q-learning results with request_rate = 1000 and delta = 100 (total_cost) 
df7 = pd.read_csv('sarsa_cache1000_request_rate1000_nb_interval40000interval_size1delta100_q_learning_NN.csv', header=0)
plt.figure()
df7['Nominal_Cost'].rolling(window=600).mean().plot()
df7['Best_Cost'].rolling(window=600).mean().plot()
df7['Cost_First'].rolling(window=600).mean().plot()
df7['Total_Cost'].rolling(window=600).mean().plot()
plt.legend(['Nominal_Cost', 'Best_Cost','Cost_First','Total_Cost'])
plt.title("Q-learning with request_rate = 1000 and delta = 100 (No normalizations)")
plt.xlabel("Time")

plt.show()