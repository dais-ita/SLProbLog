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


from unittest import TestCase
from BetaDistribution.BetaDistribution import BetaDistribution
import mpmath

class TestBetaDistribution(TestCase):
    def setUp(self):
        self.a = BetaDistribution(0.3, 0.3)
        self.b = BetaDistribution(0.4, 0.2)
        self.la = BetaDistribution("0.33333333333333", "0.2")

    def test_sum(self):
        self.assertTrue(mpmath.almosteq(self.a.mean()+self.b.mean(), (self.a.sum(self.b)).mean()))

    def test_product(self):
        self.assertTrue(mpmath.almosteq(self.a.mean() * self.b.mean(), (self.a.product(self.b)).mean()))

    def test_negate(self):
        self.assertTrue(mpmath.almosteq(1 - self.a.mean(), self.a.negate().mean()))

    def test_conditioning(self):
        self.assertTrue(mpmath.almosteq(self.a.mean() / self.b.mean(), (self.a.conditioning(self.b)).mean()))

    def test_repr(self):
        self.assertEqual(self.a.__repr__(), 'b(0.3,0.3)')

    def test_repr_long(self):
        self.assertEqual(self.la.__repr__(), 'b(0.33333333333333,0.2)')
