import math
import random

import matplotlib.pyplot as plt
import numpy as np
from regression import nadaraya_watson
from regression import Kernel
import regression

x = np.arange(0, 10, 1)
y = np.sin(x)

h = 1.5

result = np.zeros(y.size)
for i in range(y.size):
	result[i] = nadaraya_watson(y[i], x, y, h=h, kernel=Kernel.gaussian)

print("Nadaraya-Watson with K gaussian")
print("SSE %r" % regression.sse(y, result))

result1 = np.zeros(y.size)
for i in range(y.size):
	result1[i] = nadaraya_watson(y[i], x, y, h=h, kernel=Kernel.biquadratic)

print("Nadaraya-Watson with K biquadratic")
print("SSE %r" % regression.sse(y, result1))

result2 = regression.lowess_method(x, y, 2, h=h, kernel=Kernel.biquadratic)

print("LOWESS with K biquadratic")
print("SSE %r" % regression.sse(y, result2))

plt.scatter(x, y, label="Input data")
plt.plot(x, result, label='Gaussian Kernel', color='red')
plt.plot(x, result1, label='Biquadratic Kernel', color='blue')
plt.plot(x, result2, label='LOWESS Biquadratic Kernel', color='green')

plt.legend(bbox_to_anchor=(1.017, -0.1))
plt.tight_layout(pad=1.7)

plt.title("Nadaraya-Watson with K gauss")
plt.show()
