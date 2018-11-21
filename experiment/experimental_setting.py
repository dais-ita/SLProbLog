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
from SLProbLog.SLProbLog import SLProbLog
from itertools import product
import numpy.random
import pickle
import datetime
import math
from string import Formatter,Template
from problog import get_evaluatable
from problog.program import PrologString
import mpmath


class Node:
    """
    Node in a Bayesian network
    """

    def __init__(self, name):
        self._name = name
        self._parents = []
        self._children = []

    def get_name(self):
        return self._name

    def get_parents(self):
        return self._parents

    def add_parent(self, node):
        self._parents.append(node)

    def add_children(self, node):
        self._children.append(node)

    def get_children(self):
        return self._children

    def get_problog_name(self):
        return "n" + self._name

    def __repr__(self):
        return "n" + self._name

class Graph:
    """
    Data structure to represent a Bayesian network
    """

    def __init__(self):
        self._nodes = {}

    def storeFilename(self, fn):
        self.filename = fn

    def getNodes(self):
        return self._nodes

    def getFilename(self):
        return self.filename

    def add_edge(self, edge):
        for n in edge:
            if n not in self._nodes:
                self._nodes[n] = Node(n)

        self._nodes[edge[1]].add_parent(self._nodes[edge[0]])
        self._nodes[edge[0]].add_children(self._nodes[edge[1]])

    def __repr__(self):
        return str(self.__dict__)

    def get_problog_string(self):
        ret = ""
        end_loc = []
        query_loc = []
        p_index = 0

        self.semanticsprobs = {}
        self.semanticsevidences = {}
        for n in self._nodes:

            if len(self._nodes[n].get_parents()) + len(self._nodes[n].get_children()) == 1:
                end_loc.append(n)
            else:
                query_loc.append(n)

            if not self._nodes[n].get_parents():
                ret += "${p" + str(p_index) + "}::" + str(self._nodes[n].get_problog_name()) + ".\n"
                self.semanticsprobs[self._nodes[n]] = "p"+str(p_index)
                p_index += 1
            else:
                self.semanticsprobs[self._nodes[n]] = []
                for parents in product((False, True), repeat=len(self._nodes[n].get_parents())):
                    ret += "${p" + str(p_index) + "}::" + str(self._nodes[n].get_problog_name()) + " :- "
                    self.semanticsprobs[self._nodes[n]].append("p"+str(p_index))
                    p_index += 1
                    for i in range(len(parents)):
                        if not parents[i]:
                            ret += "\+"
                        ret += str(self._nodes[n].get_parents()[i].get_problog_name())

                        if i < len(parents) - 1:
                            ret += ", "
                    ret += ".\n"

        evidencenum = 0
        for n in end_loc:
            ret += "evidence("+str(self._nodes[n].get_problog_name()) + ", ${e" + str(evidencenum) + "}).\n"
            self.semanticsevidences[self._nodes[n]] = "e"+str(evidencenum)
            evidencenum += 1

        self.querynames = []
        for n in query_loc:
            ret += "query(" + str(self._nodes[n].get_problog_name()) + ").\n"
            self.querynames.append(str(self._nodes[n].get_problog_name()))
        return ret

class ProbProblog:
    """
    Given a template problog string, e.g.
    ${p1}::stress(X) :- person(X).
    evidence(..., ${e1}).
    randomly allocates values for those parameters.

    In addition, it servers as wrapper for ProbLog (function run)
    """
    def __init__(self, problogstring, network = None):
        self.network = network
        self.problogstring = problogstring
        self.keys = [ele[1] for ele in Formatter().parse(self.problogstring) if ele[1]]

        self.probabilities = {}
        self.evidences = {}
        for k in self.keys:
            if "e" in k:
                self.evidences[k] = ("true" if numpy.random.uniform(0, 1) < 0.5 else "false")
            else:
                self.probabilities[k] = numpy.random.uniform(0, 1)

    def getProbabilities(self):
        return self.probabilities

    def getEvidences(self):
        return self.evidences

    def getStructure(self):
        return self.problogstring

    def getProblogProgram(self):
        substitiutions = self.probabilities.copy()
        substitiutions.update(self.evidences)
        s = Template(self.problogstring).safe_substitute(substitiutions)
        return s

    def run(self):
        """
        Problog wrapper
        """
        res = {}
        for k, v in (get_evaluatable().create_from(PrologString(self.getProblogProgram())).evaluate()).items():
            res[str(k)] = v

        ret = {}
        for k in sorted(res):
            ret[k] = res[k]

        return ret

class Opinion():
    """
    Utility function to store a 4-tuple representing a SL opinion and outputting it
    """
    def getBelief(self):
        return mpmath.nstr(self._belief, mpmath.mp.dps)

    def getDisbelief(self):
        return mpmath.nstr(self._disbelief, mpmath.mp.dps)

    def getUncertainty(self):
        return mpmath.nstr(self._uncertainty, mpmath.mp.dps)

    def getBase(self):
        return mpmath.nstr(self._base, mpmath.mp.dps)

    def __init__(self, b, d, u=None, a=None):
        self._belief = mpmath.mpf(b)
        self._disbelief = mpmath.mpf(d)

        if u is None:
            self._uncertainty = mpmath.mpf(1 - self._belief - self._disbelief)
        else:
            self._uncertainty = mpmath.mpf(u)

        if a is None:
            self._base = mpmath.mpf("1/2")
        else:
            self._base = mpmath.mpf(a)

class DistProbLog:
    """
    Given a ProbProblog object, this class samples the randomly chosen probabilities ntrain times in order to then
    derive beta distributions
    """
    def __init__(self, bnet, ntrain=10):
        self.bn = bnet

        self.samples = {}
        self.opinions = {}
        for p in bnet.getProbabilities():
            self.samples[p] = []

            for i in range(ntrain):
                self.samples[p].append(1 if numpy.random.uniform(0, 1) < bnet.getProbabilities()[p] else 0)

            rcount = sum(self.samples[p])
            scount = ntrain - rcount
            self.opinions[p] = Opinion(rcount / (rcount + scount + 2), scount / (rcount + scount + 2))

    def get_program(self):
        """
        Substitute the various probabilities signposts with SL opinions
        """
        substitutions = {}
        for k in self.opinions:
            substitutions[k] = "w(%s,%s,%s,%s)" % (self.opinions[k].getBelief(),
                                                   self.opinions[k].getDisbelief(),
                                                   self.opinions[k].getUncertainty(),
                                                   self.opinions[k].getBase())
        substitutions.update(self.bn.evidences)
        return Template(self.bn.problogstring).safe_substitute(substitutions)


class Experiment():
    """
    Class collecting methods for running experiments
    """

    def loadExperiment(picklefile):
        """
        Load from a pickle file
        """
        return pickle.load(open(picklefile, "rb"))

    def __init__(self, picklefile = None):
        """
        Loads from the pickle if passed as parameter, otherwise just creates empty attributes
        """
        if picklefile:
            self = pickle.load(open(picklefile, "rb"))

        self._vec_real = None
        self._vec_sl = None
        self._vec_likelihood = None
        self._vec_moments = None
        self._vec_purebn = None
        self._vec_sbn = None

        self._Nmonte = None
        self._Nnetworks = None
        self._sampleBeta = None
        self._sampleMontecarlo = None

        self._name = None
        self._problogstring = None

        self._is_this_a_bn = None

    def setup(self, name, content, Nmonte = 10, Nnetworks = 100, sampleBeta = [10], bn=True):
        """
        Storage of attributes
        :param name: name of this experiment: anything
        :param content: file with Bayesian network specified or string otherwise
        :param Nmonte: how often we want to randomly assign probabilities to the template problog string
        :param Nnetworks: how many networks we want to generate
        :param sampleBeta: how many samples to use for create SL opinions
        :param bn: is this a Bayesian network?
        :return:
        """

        self._Nmonte = Nmonte
        self._Nnetworks = Nnetworks
        self._sampleBeta = sampleBeta

        self._name = name
        self._problogstring = None

        self._is_this_a_bn = bn

        self._is_this_a_bn = bn
        if bn:
            self.net = Graph()

            with open(content) as f:
                for line in f:
                    self.net.add_edge(line.rstrip('\n').split(" "))
            self.net.storeFilename(content)
            self._problogstring = self.net.get_problog_string()
        else:
            if not isinstance(content,str):
                raise Exception("Todo")

            self.net = None

            self._problogstring = content

    def _expected_value(self,l):
        """
        Returns the expected value of a SL opinion
        """
        return l[0] + l[2] * l[3]

    def _compute_error(self, one, two):
        """
        Compute the distance between the two values
        """
        res = 0
        items = 0
        if len(one) != len(two):
            print(len(one))
            print(len(two))
            raise Exception("wrong length")

        for i in range(len(one)):
            for k, x in one[i].items():
                w = two[i][k]

                p = None
                if isinstance(x, list):
                    p = self._expected_value(x)
                else:
                    p = x

                if isinstance(w, list):
                    res += (p - float(self._expected_value(w))) ** 2
                else:
                    res += (p - float(w)) ** 2
                items += 1
        return math.sqrt(float(res) / float(items))

    def _expected_error(self, listw):
        """
        Computed the expected error of a SL opinion
        """
        res = 0
        items = 0
        for w in listw:
            for k, v in w.items():
                items += 1
                res += (float(self._expected_value(v)) * (1.0-float(self._expected_value(v))) * float(v[2]) / (2.0 + float(v[2])))

        return math.sqrt(float(res) / float(items))

    def run(self):
        """
        Run the experiment with the given setup
        """
        self._vec_real = []
        self._vec_sl = []
        self._vec_sl_beta = []

        self.bns = []

        Nruns = self._Nmonte * self._Nnetworks

        for i in range(Nruns):

            sys.stdout.write("\r%d%%" % int(i/Nruns*100))
            sys.stdout.flush()

            b = None
            if b is None or i % self._Nmonte:
                b = ProbProblog(self._problogstring, self.net)
                self.bns.append(b)

            self._vec_real.append(b.run())

            for samples in self._sampleBeta:
                sb = DistProbLog(b, samples)
                self._vec_sl.append(SLProbLog(sb.get_program(), True).run_SL())
                self._vec_sl_beta.append(SLProbLog(sb.get_program(), True).run_beta())

        print("")
        self._store()


    def _store(self):
        """
        Save the pickle file
        """
        now = datetime.datetime.now()
        self._filename = self._name + "-%s-%s-%s-%s-%s" % (now.year, now.month, now.day, now.hour, now.minute)

        pickle.dump(self, open(self._filename + ".pickle", "wb"))


    def analise(self, graphs = True):
        """
        Analyse the results and print them to screen
        """
        strreal = "& & A & "
        strpred = "& & P & "

        if self._vec_real is None or self._vec_sl is None or self._vec_sl_beta is None:
            raise Exception("Run first")


        strreal += "%.4f & " % self._compute_error(self._vec_real, self._vec_sl_beta)
        strpred += "%.4f & " % self._expected_error(self._vec_sl_beta)

        strreal += "%.4f & " % self._compute_error(self._vec_real, self._vec_sl)
        strpred += "%.4f & " % self._expected_error(self._vec_sl)

        strreal = strreal[:-2]
        strreal += "\\\\"

        strpred = strpred[:-2]
        strpred += "\\\\"

        print(strreal)
        print(strpred)