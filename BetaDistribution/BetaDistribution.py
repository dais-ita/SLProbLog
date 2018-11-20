"""
Copyright (c) 2018 Federico Cerutti <CeruttiF@cardiff.ac.uk>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import mpmath

EPSILON = 10e-4

def from_sl_opinion(wb, W = 2):
    prior = mpmath.mpf(W)

    [belief, disbelief, uncertainty, base] = wb

    if mpmath.almosteq(mpmath.mpf("0"), uncertainty, EPSILON):
        uncertainty = mpmath.mpf(EPSILON)

    if mpmath.almosteq(mpmath.mpf("0"), disbelief, EPSILON):
        disbelief = mpmath.mpf(EPSILON)

    if mpmath.almosteq(mpmath.mpf("0"), belief, EPSILON):
        belief = mpmath.mpf(EPSILON)

    alpha = prior / uncertainty * belief + prior * base
    beta = prior / uncertainty * disbelief + prior * (1 - base)

    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * alpha + beta + 1)

    return BetaDistribution(mean, variance)

class BetaDistribution():

    def __init__(self, m, v):
        self._epsilon = EPSILON
        self._ZERO = mpmath.mpf("0")
        self._ONE = mpmath.mpf("1")
        self._mu = mpmath.mpf(m)
        self._var = mpmath.mpf(v)

    def is_complete_belief(self):
        if mpmath.almosteq(self.mean(), self._ONE, self._epsilon):
            return True
        return False

    def mean(self):
        return self._mu

    def variance(self):
        return self._var

    def strength(self):
        var = self.variance()
        if mpmath.almosteq(var, 0, self._epsilon):
            var = mpmath.mpf(self._epsilon)

        return (self.mean() * (1 - self.mean())) / var - 1

    def alpha(self):
        return max(mpmath.mpf(self._epsilon), self.mean() * self.strength())

    def beta(self):
        return max(mpmath.mpf(self._epsilon), (1 - self.mean()) * self.strength())

    def sum(self, Y):
        mean = self.mean() + Y.mean()
        var = self.variance() + Y.variance()

        var = min(var, mean ** 2 * (1.0 - mean) / (1.0 + mean), (1.0 - mean) ** 2 * mean / (2 - mean))

        return BetaDistribution(mean,var)

    def product(self, Y):
        mean = self.mean() * Y.mean()
        var = self.variance() * Y.variance() + \
              self.variance() * (Y.mean())**2 + Y.variance() * (self.mean()) ** 2

        var = min(var, mean ** 2 * (1.0 - mean) / (1.0 + mean), (1.0 - mean) ** 2 * mean / (2 - mean))

        return BetaDistribution(mean, var)

    def negate(self):
        if not 0 <= self.mean() <= 1:
            raise Exception("Error with negation: [%f, %f]", (self.mean(), self.variance()))
        return BetaDistribution(1.0 - self.mean(), self.variance())

    def conditioning(self, Y):
        mean = min(1.0-1e-6, self.mean() / Y.mean())

        muneg = Y.mean() - self.mean() #+ Y.mean() * self.mean()
        varneg = Y.variance() - self.variance()

        product = self.product(BetaDistribution(muneg, varneg))

        if self._mu <= 0:
            self._mu = mpmath.mpf(1e-10)

        if muneg <= 0:
            muneg = mpmath.mpf(1e-10)

        var = mean**2 * (1.0-mean)**2 * ((self.variance() / (self._mu ** 2)) + (varneg / (muneg ** 2)) - 2 * (product.variance() / (self._mu * muneg)))

        var = min(var, mean ** 2 * (1.0 - mean) / (1.0 + mean), (1.0 - mean) ** 2 * mean / (2 - mean))

        return BetaDistribution(mean, var)

    def __repr__(self):
        return "b(%s,%s)" % (mpmath.nstr(self.mean(), mpmath.mp.dps), mpmath.nstr(self.variance(), mpmath.mp.dps))

    def to_sl_opinion(self, a = 1/2, W=2):
        rx = max(mpmath.mpf(0), self.alpha() - a * W)
        sx = max(mpmath.mpf(0), self.beta() - (1-a) * W)
        return [(rx / (rx + sx + W)), (sx / (rx + sx + W)), (W / (rx + sx + W)), mpmath.mpf(a)]