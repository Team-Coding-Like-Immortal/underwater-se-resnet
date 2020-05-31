import matplotlib.pyplot as plt
import numpy as np
import math as m

plt.figure(figsize=(8, 8))
t1 = (1.864 + 1.841 + 1.868 + 1.965 + 2.030) / 5
t2 = (2.284 + 2.258 + 2.349 + 2.330 + 2.332) / 5
t3 = (2.592 + 2.550 + 2.579 + 2.618 + 2.574) / 5
t4 = (3.308 + 3.342 + 3.342 + 3.257 + 3.255) / 5
t5 = (4.398 + 4.180 + 4.090 + 3.735 + 4.635) / 5
cosT = np.array([0.731, 0.766, 0.809, 0.839, 0.875])
_t = np.array([t1, t2, t3, t4, t5])
_v = 35.5 / _t
x = _v / cosT
print(x)
y = np.array([0.963, 0.839, 0.737, 0.649, 0.560])
print(y)

plt.plot(x,y,color='b')
plt.scatter(x,y,c='r',marker='o')
plt.text(25.37814146,0.933,s=(25.378, 0.963))
plt.text(20.0574,0.839,s=(20.057, 0.839))
plt.text(16.99114651,0.727,s=(16.991, 0.737))
plt.text(12.81879439,0.649,s=(12.818, 0.649))
plt.text(9.64241576,0.554,s=(9.642, 0.560))
# plt.show()
plt.savefig('result.png')
# print(t1,t2,t3,t4,t5)
