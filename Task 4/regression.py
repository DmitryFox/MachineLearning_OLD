import math
import numpy as np


class Kernel:
    @staticmethod
    def rectangular(u):
        return 1. / 2. if math.fabs(u) <= 1. else 0.

    @staticmethod
    def triangular(u):
        return 1. - math.fabs(u) if math.fabs(u) <= 1. else 0.

    @staticmethod
    def epanechnikov(u):
        return 3. / 4. * (1. - math.pow(u, 2.)) if math.fabs(u) <= 1. else 0.

    @staticmethod
    def quartic(u):
        return 15. / 16. * math.pow(1. - math.pow(u, 2.), 2.) if math.fabs(u) <= 1. else 0.

    @staticmethod
    def triweight(u):
        return 35. / 32. * math.pow(1. - math.pow(u, 2.), 3.) if math.fabs(u) <= 1. else 0.

    @staticmethod
    def gaussian(u):
        return 1. / math.sqrt(2. * math.pi) * math.exp(-1. / 2. * math.pow(u, 2.))


class Metric:
    @staticmethod
    def euclidean(p, q):
        return math.sqrt(math.pow(p - q, 2.))


def nadaraya_watson(value, x, y, h, kernel=Kernel.gaussian, metric=Metric.euclidean):
    size = x.size
    weight = np.zeros(size)
    for i in range(size):
        weight[i] = kernel(metric(value, x[i]) / h)
    return sum(weight * y) / sum(weight)


def lowess_method(x, y, iteration, h, kernel=Kernel.gaussian, metric=Metric.euclidean):
    size = x.size
    gamma = np.ones(size)
    result = np.zeros(size)
    for step in range(iteration):
        for t in range(size):
            numerator = 0.
            denominator = 0.
            for i in range(size):
                numerator += y[i] * gamma[i] * kernel(metric(x[i], x[t]) / h)
                denominator += gamma[i] * kernel(metric(x[i], x[t]) / h)
            result[t] = numerator / denominator if denominator != 0. else 0.
        for i in range(size):
            gamma[i] = kernel(math.fabs(result[i] - y[i]))
    return result


def sse(y_original, y_result):
    return np.sum(pow(y_original - y_result, 2))

