#!/usr/bin/env python3
import numpy as np
import sys

def read_matrix_file(filename):
    with open(filename) as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    size = int(lines[0])
    data = []
    for line in lines[1:]:
        data += [float(x) for x in line.split()]
    
    if len(data) >= 2 * size * size:
        A = np.array(data[:size*size]).reshape(size, size)
        B = np.array(data[size*size:2*size*size]).reshape(size, size)
        return A, B, size
    return None, None, size

def read_result(filename):
    with open(filename) as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    size = int(lines[0])
    matrix = []
    for line in lines[1:size+1]:
        matrix.append([float(x) for x in line.split()])
    return np.array(matrix)

if len(sys.argv) != 3:
    print("Usage: python verify.py <input_file> <result_file>")
    sys.exit(1)

A, B, size = read_matrix_file(sys.argv[1])
result = read_result(sys.argv[2])

expected = A @ B
diff = np.max(np.abs(result - expected))

print(f"Size: {size}x{size}")
print(f"Max difference: {diff:.2e}")
print("PASS" if diff < 1e-6 else "FAIL")