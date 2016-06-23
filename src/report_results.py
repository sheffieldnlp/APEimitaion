import sys
import os
import pprint
import numpy as np

folder = sys.argv[1]
files = os.listdir(folder)
results = {}
best_results = {}
for filename in files:
    with open(os.path.join(folder, filename)) as f:
        dev_results = []
        test_results = []
        dev_mode = False
        test_mode = False
        report_mode = False
        for line in f:
            if line.strip() == 'RESULTS DEV':
                dev_mode = True
                test_mode = False
            if line.strip() == 'RESULTS TEST':
                dev_mode = False
                test_mode = True
            if 'YOUR RESULTS' in line:
                report_mode = True
            if 'bad times' in line:
                if report_mode:
                    if dev_mode:
                        dev_results.append(float(line.split()[-1]))
                    if test_mode:
                        test_results.append(float(line.split()[-1]))
                    report_mode = False
        results[filename] = (dev_results, test_results)
        best_dev_index = np.argmax(dev_results)
        best_results[filename] = (best_dev_index,
                                  dev_results[best_dev_index],
                                  test_results[best_dev_index])


pprint.pprint(best_results)
