import sys

from SLProbLog.SLProbLog import SLProbLog

if __name__ == '__main__':
    p = ""
    with open(sys.argv[1], 'r') as f:
        p = f.read()

    print(SLProbLog(p).run_beta())