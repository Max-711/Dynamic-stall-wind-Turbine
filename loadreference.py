import numpy as np
import os

def load_reference(path="bladed_reference.csv"):
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or line[0].isalpha():
            continue
        rows.append([float(v) for v in line.split(",")])
    d = np.array(rows)
    return d[:, 0], np.radians(d[:, 1]), d[:, 2], d[:, 3]


