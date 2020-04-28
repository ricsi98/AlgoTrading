import numpy as np
import matplotlib.pyplot as plt

def getNext(prev, volatility):
    return abs(prev + volatility * (np.random.uniform() - 0.5))

arr = [100]
for i in range(50000):
    arr.append(getNext(arr[-1], abs(np.sin(float(i)/1000))))

plt.plot(arr)
plt.show()