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


import sys
sys.path.append('../')

from experiment.experimental_setting import Experiment
e = Experiment()


model="""
${p1}::stress(X) :- person(X).
${p2}::influences(X,Y) :- person(X), person(Y).

smokes(X) :- stress(X).
smokes(X) :- friend(X,Y), influences(Y,X), smokes(Y).

${p5}::asthma(X) :- smokes(X).

person(1).
person(2).
person(3).
person(4).

friend(1,2).
friend(2,1).
friend(2,4).
friend(3,2).
friend(4,2).

evidence(smokes(2),true).
evidence(influences(4,2),false).

query(smokes(1)).
query(smokes(3)).
query(smokes(4)).
query(asthma(1)).
query(asthma(2)).
query(asthma(3)).
query(asthma(4)).
"""

e.setup("smoker-evidence-Nins10", model, 10, 100, [10], bn=False)

e.run()

e.analise()

e.setup("smoker-evidence-Nins50", model, 10, 100, [50], bn=False)

e.run()

e.analise()

e.setup("smoker-evidence-Nins100", model, 10, 100, [100], bn=False)

e.run()

e.analise()