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
from SLProbLog.SLProbLog import SLProbLog
import mpmath


class TestSLProbLog(TestCase):

    def setUp(self):
        self.smallprogram = """
b(0.6, 0.0002)::asthma(X) :- smokes(X).
smokes(bill).
query(asthma(bill)).
"""

    def test_sl_operators_beta_run(self):
        p = SLProbLog(self.smallprogram)
        self.assertEqual(str(p.run_beta()), "{'asthma(bill)': b(0.6,0.0002)}")

    def test_sl_operators_sl_run(self):
        p = SLProbLog(self.smallprogram)
        r = p.run_SL()

        self.assertTrue(mpmath.almosteq(r['asthma(bill)'].mean(), 0.6, 0.001))