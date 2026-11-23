"""
pca_verify.py
Manual PCA projection of one 2D point onto PC1 (1D representation).

Steps:
  1. Standardize each feature:  Z = (x - mu) / sigma
  2. Build the 1x2 standardized row vector
  3. Multiply by the 2x1 PC1 column vector (matrix multiplication)

NOTE: Only mu1 = 25 and sigma1 = 11.18 were given in the task.
Everything marked PLACEHOLDER below is an assumed value -- replace it
with your real data and re-run.
"""

import numpy as np

# ---------------- Inputs ----------------
# Given in the task
mu1, sigma1 = 25.0, 11.18

# PLACEHOLDERS (not provided in the task) -- edit these
x1, x2 = 30.0, 50.0          # PLACEHOLDER: the data point (x1, x2)
mu2, sigma2 = 40.0, 15.0     # PLACEHOLDER: feature 2 mean and std
pc1 = np.array([[0.7071],    # PLACEHOLDER: top eigenvector (eigenvalue max),
                [0.7071]])   # as a 2x1 column vector

# ---------------- Step 1: standardize ----------------
z1 = (x1 - mu1) / sigma1
z2 = (x2 - mu2) / sigma2

# ---------------- Step 2: 1x2 standardized vector ----------------
z = np.array([[z1, z2]])     # shape (1, 2)

# ---------------- Step 3: project (1x2 @ 2x1 -> 1x1) ----------------
projection = z @ pc1         # shape (1, 1)
y = projection.item()

# ---------------- Output ----------------
print("Data point x            :", (x1, x2))
print(f"Feature 1  (mu, sigma)  : ({mu1}, {sigma1})")
print(f"Feature 2  (mu, sigma)  : ({mu2}, {sigma2})")
print(f"z1 = ({x1} - {mu1}) / {sigma1} = {z1:.4f}")
print(f"z2 = ({x2} - {mu2}) / {sigma2} = {z2:.4f}")
print("Standardized vector z   :", z.round(4), "shape", z.shape)
print("PC1 (2x1)               :", pc1.T.round(4), "(transposed for display) shape", pc1.shape)
print(f"Projection z @ PC1      : {y:.4f}")
print(f"\nFinal 1D projected coordinate = {y:.4f}")
