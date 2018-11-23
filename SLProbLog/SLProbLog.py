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

from problog.evaluator import Semiring
from problog.engine import DefaultEngine
from problog.program import PrologString
from problog import get_evaluatable
import mpmath

EPSILON = 10e-100
mpmath.dps = 200

def from_sl_opinion(wb, W = 2):
    prior = mpmath.mpf(W)

    [belief, disbelief, uncertainty, base] = wb

    if mpmath.almosteq(mpmath.mpf("0"), uncertainty, EPSILON):
        uncertainty = mpmath.mpf(EPSILON)

    if mpmath.almosteq(mpmath.mpf("0"), disbelief, EPSILON):
        disbelief = mpmath.mpf(EPSILON)

    if mpmath.almosteq(mpmath.mpf("0"), belief, EPSILON):
        belief = mpmath.mpf(EPSILON)

    mean = belief + uncertainty * base
    sx = prior / uncertainty
    variance = mean * (1- mean) / (sx + 1)

    return BetaDistribution(mean, variance)


def moment_matching(b):
    m = b.mean()
    v = b.variance()
    if v == 0:
        var = mpmath.mpf(1e-10)
    else:
        var = mpmath.mpf(v)

    mean = min(mpmath.mpf(1), max(mpmath.mpf(0), mpmath.mpf(m)))

    sx = ((mean * (1 - mean)) / var - 1)

    return BetaDistribution(mean, (mean * (1-mean) / (sx + 1) ))


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

    def mean_str(self):
        return mpmath.nstr(self.mean(), mpmath.mp.dps)

    def variance_str(self):
        return mpmath.nstr(self.variance(), mpmath.mp.dps)

    def to_sl_opinion(self, a = 1/2, W=2):
        rx = max(mpmath.mpf(0), self.alpha() - a * W)
        sx = max(mpmath.mpf(0), self.beta() - (1-a) * W)
        return [(rx / (rx + sx + W)), (sx / (rx + sx + W)), (W / (rx + sx + W)), mpmath.mpf(a)]

class SLSemiring(Semiring):

    def parse(self, w):
        start = w.find('(') + 1
        end = w.find(')')
        return [mpmath.mpf(x) for x in w[start:end].replace(" ","").split(',')]

    def one(self):
        return "w(1.0, 0.0, 0.0, 1.0)"

    def zero(self):
        return "w(0.0, 1.0, 0.0, 0.0)"

    def plus(self, x, y):
        [b1,d1,u1,a1] = self.parse(x)
        [b2,d2,u2,a2] = self.parse(y)
        u = (a1 * u1 + a2 * u2) / (a1 + a2)
        d = max(0.0, (a1 * (d1 - b2) + a2 * (d2 - b1)) / (a1 + a2))
        b = min(b1 + b2, 1.0)
        a = min(a1 + a2, 1.0)
        return "w(%s,%s,%s,%s)" % (str(b), str(d), str(u), str(a))

    def times(self, x, y):
        [b1, d1, u1, a1] = self.parse(x)
        [b2, d2, u2, a2] = self.parse(y)
        a = a1 * a2
        b = b1 * b2 + ((1 - a1) * a2 * b1 * u2 + a1 * (1 - a2) * u1 * b2) / (1 - a1 * a2)
        u = u1 * u2 + ((1 - a2) * b1 * u2 + (1 - a1) * u1 * b2) / (1 - a1 * a2)
        d = min(1, d1 + d2 - d1 * d2)
        return "w(%s,%s,%s,%s)" % (str(b), str(d), str(u), str(a))

    def negate(self, a):
        [b1, d1, u1, a1] = self.parse(a)
        return "w(%s,%s,%s,%s)" % (str(d1), str(b1), str(u1), str(1 - a1))

    def value(self, a):
        return str(a)

    def normalize(self, x, z):

        if z == self.one():
            return x

        [b1, d1, u1, a1] = self.parse(x)
        [b2, d2, u2, a2] = self.parse(z)
        e1 = b1 + u1*a1
        e2 = b2+ u2 * a2

        if not ((a1<=a2) and (d1>=d2) and (b1*(1-a1)*a2*(1-d2) >= a1*(1-a2)*(1-d1)*b2) and (u1*(1-a1)*(1-d2)>=u2*(1-a2)*(1-d1)) and a2!=0 ):
            return "w(%s,%s,%s,%s)" % (str(0.0), str(0.0), str(1.0), str(0.5))
        else:
            a = a1/a2
            b = 0.0
            d = 0.0
            u = 0.0
            if e1 == 0:
                d = 1.0
            elif a==1:
                b = 1.0
            else:
                e = e1 / e2
                d = min(max(0, (d1 - d2) / (1 - d2)), 1)
                u = min(max(0, (1 - d - e) / (1 - a)), 1)
                b = min(max(0, (1 - d - u)), 1)
            return "w(%s,%s,%s,%s)" % (str(b), str(d), str(u), str(a))

    def is_dsp(self):
        return True

class BetaSemiring(Semiring):

    def parse(self, w):
        start = str(w).find('(') + 1
        end = str(w).find(')')
        parsed = [mpmath.mpf(x) for x in str(w)[start:end].replace(" ","").split(',')]
        return BetaDistribution(parsed[0], parsed[1])

    def one(self):
        return "b(1.0,0.000000001)"

    def zero(self):
        return "b(0.0,0.000000001)"

    def plus(self, a, b):
        wa = self.parse(a)
        wb = self.parse(b)
        return self._to_str(wa.sum(wb))

    def times(self, a, b):
        wa = self.parse(a)
        wb = self.parse(b)
        return self._to_str(wa.product(wb))

    def negate(self, a):
        wa = self.parse(a)
        return self._to_str(wa.negate())

    def value(self, a):
        return str(a)

    def _to_str(self, r):
        wr = self.parse(r)
        return wr.__repr__()

    def normalize(self, a, z):
        wa = self.parse(a)
        wz = self.parse(z)

        if wz.is_complete_belief():
            return a
        return self._to_str(wa.conditioning(wz))

    def is_dsp(self):
        return True



class SLProbLog:

    def __init__(self, program, sloutput = False):
        self._slproblog_program = program
        self._slout = sloutput

    def _convert_input(self, to_sl = False, to_beta = False):
        if to_sl and to_beta:
            raise Exception("Cannot convert both to SL and to Beta")

        lines = self._slproblog_program.splitlines()
        newlines = []
        for l in lines:
            label_prolog = l.split("::")
            if len(label_prolog) == 1:
                newlines.append(l)
            elif len(label_prolog) == 2:
                if to_sl:
                    if label_prolog[0][0] == "b":
                        b = BetaSemiring().parse(label_prolog[0])
                        wb = b.to_sl_opinion()
                        newlines.append("w(%s,%s,%s,%s)::%s" % (mpmath.nstr(wb[0]),
                                                                mpmath.nstr(wb[1]),
                                                                mpmath.nstr(wb[2]),
                                                                mpmath.nstr(wb[3]),
                                                                label_prolog[1]))

                    elif label_prolog[0][0] == "w":
                        newlines.append(l)

                    else:
                        raise Exception("Problem with this line: %s" % (l))


                elif to_beta:
                    if label_prolog[0][0] == "w":
                        w = SLSemiring().parse(label_prolog[0])
                        bw = from_sl_opinion(w)
                        newlines.append("%s::%s" % (repr(bw), label_prolog[1]))

                    elif label_prolog[0][0] == "b":
                        newlines.append(l)

                    else:
                        raise Exception("Problem with this line: %s" % (l))

            else:
                raise Exception("Problem with this line: %s" % (l))

        return "\n".join(newlines)

    def _convert_output(self, res, to_sl = False, to_beta = False):
        if to_sl and to_beta:
            raise Exception("Cannot convert both to SL and to Beta")

        ret = {}
        for k, v in res.items():
            if to_sl and isinstance(v, BetaDistribution):
                ret[k] = v.to_sl_opinion()
            elif to_beta and isinstance(v, list):
                ret[k] = from_sl_opinion(v)

        return ret


    def run_SL(self):
        res = self._run_sl_operators_on_semiring(SLSemiring(), self._convert_input(to_sl = True))
        if self._slout:
            return res
        return self._convert_output(res, to_beta = True)

    def run_beta(self):
        res = self._run_sl_operators_on_semiring(BetaSemiring(), self._convert_input(to_beta = True))
        if self._slout:
            return self._convert_output(res, to_sl=True)
        return res


    def _run_sl_operators_on_semiring(self, givensemiring, program = None):
        engine = DefaultEngine()
        if program == None:
            program = self._slproblog_program

        db = engine.prepare(PrologString(program))
        semiring = givensemiring
        knowledge = get_evaluatable(None, semiring=semiring)
        formula = knowledge.create_from(db, engine=engine, database=db)
        res = formula.evaluate(semiring=semiring)

        ret = {}
        for k, v in res.items():
            if isinstance(semiring, BetaSemiring):
                ret[k] = moment_matching(semiring.parse(v))
            else:
                ret[k] = semiring.parse(v)

        return self._order_dicts(ret)


    def _order_dicts(self, dicinput):
        res = {}
        for k, v in dicinput.items():
            res[str(k)] = v

        ret = {}
        for k in sorted(res):
            ret[k] = res[k]

        return ret