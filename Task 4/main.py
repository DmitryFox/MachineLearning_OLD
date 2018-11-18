import random

import matplotlib.pyplot as plt
import numpy as np
from regression import nadaraya_watson
from regression import Kernel
import regression

x = np.arange(1, 50, 1)
y = x * np.random.random(x.size)
y_orig = x

h = 8.2

result = np.zeros(x.size)
for i in range(x.size):
    result[i] = nadaraya_watson(x[i], x, y, h=h, kernel=Kernel.gaussian)

result1 = regression.lowess_method(x, y, 2, h=h, kernel=Kernel.gaussian)

print("[Gaussian K] Nadaraya-Watson")
print("SSE %r" % regression.sse(y_orig, result))
print()
print("[Gaussian K] LOWESS")
print("SSE %r" % regression.sse(y_orig, result1))

plt.scatter(x, y, s=5)
plt.plot(x, result, label='[Gaussian K] Nadaraya-Watson', color='red')
plt.plot(x, result1, label='[Gaussian K] LOWESS', color='blue')

plt.legend(bbox_to_anchor=(1.017, -0.1))
plt.tight_layout(pad=1.7)

plt.title("LOWESS & Nadaraya-Watson with gaussian kernel")
plt.figure()

print("-------------------------------------")

result2 = np.zeros(x.size)
for i in range(x.size):
    result2[i] = nadaraya_watson(x[i], x, y, h=h, kernel=Kernel.quartic)

result3 = regression.lowess_method(x, y, 2, h=h, kernel=Kernel.quartic)

print("[Quartic K] Nadaraya-Watson")
print("SSE %r" % regression.sse(y_orig, result2))
print()
print("[Quartic K] LOWESS")
print("SSE %r" % regression.sse(y_orig, result3))

plt.scatter(x, y, s=5)
plt.plot(x, result2, label='[Quartic K] Nadaraya-Watson', color='red')
plt.plot(x, result3, label='[Quartic K] LOWESS', color='blue')

plt.legend(bbox_to_anchor=(1.017, -0.1))
plt.tight_layout(pad=1.7)

plt.title("LOWESS & Nadaraya-Watson with quartic kernel")
plt.show()
