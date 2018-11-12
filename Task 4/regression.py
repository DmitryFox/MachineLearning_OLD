import math
import numpy as np


class Kernel:
    @staticmethod
    def gaussian(u):
        return 1. / math.sqrt(2. * math.pi) * math.exp(-1. / 2. * math.pow(u, 2.))

    @staticmethod
    def biquadratic(u):
        return 15. / 16. * math.pow(1. - math.pow(u, 2.), 2.) if math.fabs(u) <= 1. else 0.


class Metric:
    @staticmethod
    def euclidean(q, p):
        return math.sqrt(math.pow(q - p, 2.))


def sse(y_real, y_result):
    """ return sum of squared errors. """
    size = y_real.size
    sum = 0.
    for i in range(size):
        sum += pow(y_real[i] - y_result[i], 2)
    return sum / size


def nadaraya_watson(value, x, y, h, kernel=Kernel.gaussian, metric=Metric.euclidean):
    size = x.size
    weight = np.zeros(size)
    for i in range(size):
        weight[i] = kernel(metric(value, x[i]) / h)
    return sum(weight * y) / sum(weight)


def lowess_method(x, y, iteration, h, kernel=Kernel.gaussian, metric=Metric.euclidean):
    size = x.size
    delta = np.ones(size)
    Yt = np.zeros(size)
    for step in range(iteration):
        for t in range(size):
            numerator = 0.
            denominator = 0.
            for i in range(size):
                numerator += y[i] * delta[i] * kernel(metric(x[i], x[t]) / h)
                denominator += y[i] * kernel(metric(x[i], x[t]) / h)
            Yt[t] = numerator / denominator
        d = np.abs(y - Yt)
        delta = [kernel(d[j]) for j in range(size)]
    return Yt

