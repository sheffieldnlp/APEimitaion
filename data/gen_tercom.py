import sys

FNAME = sys.argv[1]

with open(FNAME) as f:
    for i, line in enumerate(f.readlines()):
        print line.strip() + ('\t(%d)' % i)

