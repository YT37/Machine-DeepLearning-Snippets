import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dataset = pd.read_csv('AdsCTROptimisation.csv')

adsSelected = []
total = 0
for n in range(0, 10000):
    ad = random.randrange(10)
    adsSelected.append(ad)
    reward = dataset.values[n, ad]
    total = total + reward

plt.hist(adsSelected)
plt.title('Histogram Of Ads Selections')
plt.xlabel('Ads')
plt.ylabel('Number Of Times Each Ad Was Selected')
plt.show()