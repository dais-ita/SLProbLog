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

import argparse
import mpmath
from SLProbLog.SLProbLog import BetaDistribution

from SLProbLog.SLProbLog import SLProbLog

def outprint(res):
    for k,v in res.items():
        if isinstance(v, list):
            print("%-12s [%-23s %-23s %-23s %s]\t" % (k, mpmath.nstr(v[0], mpmath.mp.dps),
                                                            mpmath.nstr(v[1], mpmath.mp.dps),
                                                            mpmath.nstr(v[2], mpmath.mp.dps),
                                                            mpmath.nstr(v[3], mpmath.mp.dps)))
        elif isinstance(v, BetaDistribution):
            print("%-12s [%-23s %-23s]\t" % (k, mpmath.nstr(v.mean(), mpmath.mp.dps), mpmath.nstr(v.variance(), mpmath.mp.dps)))
        else:
            raise Exception("Unclear data: %s" % (repr(v)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input file")
    parser.add_argument("-slop", "--subjective-logic-operators", help="Use SL Operators instead of Beta-based", action="store_true")
    parser.add_argument("-slout", "--subjective-logic-output", help="Output as Subjective Logic Opinions",
                        action="store_true")


    args = parser.parse_args()

    p = ""
    with open(args.file, 'r') as f:
        p = f.read()

    if args.subjective_logic_operators:
        outprint(SLProbLog(p, args.subjective_logic_output).run_SL())
    else:
        outprint(SLProbLog(p, args.subjective_logic_output).run_beta())