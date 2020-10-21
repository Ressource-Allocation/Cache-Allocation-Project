import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('sarsa_cache1000_request_rate10_nb_interval40000interval_size1delta20.csv', header=1)
plt.plot(df['Time_hours'],df['Total_Cost'])