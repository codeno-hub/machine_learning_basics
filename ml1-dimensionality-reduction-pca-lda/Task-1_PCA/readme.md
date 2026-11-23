# Task 1: Principal Component Analysis (PCA) Projection

This module demonstrates how a 2-dimensional feature vector is compressed into a 1-dimensional scalar coordinate by projecting it onto the first Principal Component ($\text{PC}_1$)[cite: 2, 4].

---

## Files in This Directory
* `pca_verify.py` — NumPy matrix multiplication script verifying the manual projection[cite: 2].
* `pca_calculation.pdf` — Step-by-step analytical derivation[cite: 4].
* `Screenshot 2026-11-23 190606.png` — Console output verifying the 1D projected coordinate[cite: 3].

---

## Mathematical Foundations

### 1. Why Standardize First?
PCA identifies orthogonal directions of maximum variance. If features operate on different scales (for example, one ranging from $0\text{–}100$ and another from $0\text{–}10$), the feature with the larger numerical magnitude will dominate the covariance matrix artificially.

**Z-score standardization** centers every feature at a mean of $0$ with a standard deviation of $1$:

$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$

where:
* $x_i$ = raw feature value
* $\mu_i$ = feature mean across the dataset
* $\sigma_i$ = feature standard deviation across the dataset

---

### 2. Eigen-Decomposition & Principal Components
Given standardized data $\mathbf{Z}$, PCA computes the covariance matrix $\Sigma$:

$$\Sigma = \frac{1}{n - 1} \mathbf{Z}^T \mathbf{Z}$$

Solving the eigenvalue problem $\Sigma \mathbf{v} = \lambda \mathbf{v}$ yields:
* **Eigenvectors ($\mathbf{v}_k$):** Orthogonal unit vectors defining the new principal axes.
* **Eigenvalues ($\lambda_k$):** Scalars measuring the amount of variance captured along each corresponding eigenvector.

We select the eigenvector $\mathbf{v}_1$ ($\text{PC}_1$) associated with the **largest eigenvalue ($\lambda_{\max}$)** because it preserves the maximum possible variance (information) when reducing dimensionality from 2D to 1D[cite: 2, 4].

---

### 3. Linear Projection via Dot Product
To map a standardized 2D row vector $\mathbf{z} = [z_1, \; z_2]$ onto the 1D subspace spanned by unit vector $\mathbf{v}_1 = [v_{11}, \; v_{21}]^T$, we compute their inner (dot) product[cite: 2, 4]:

$$y = \mathbf{z} \cdot \mathbf{v}_1 = \begin{bmatrix} z_1 & z_2 \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{21} \end{bmatrix} = z_1 v_{11} + z_2 v_{21}$$

This operation projects the point orthogonally onto the $\text{PC}_1$ axis, compressing two coordinates into a single scalar value $y$[cite: 2, 4].

---

## Worked Numerical Example

### Input Parameters
| Parameter | Symbol | Value |
| :--- | :---: | :---: |
| Raw Data Point | $\mathbf{x} = (x_1, x_2)$ | $(30.0, \; 50.0)$[cite: 2, 4] |
| Feature 1 Mean & Std Dev | $(\mu_1, \sigma_1)$ | $(25.0, \; 11.18)$[cite: 2, 4] |
| Feature 2 Mean & Std Dev | $(\mu_2, \sigma_2)$ | $(40.0, \; 15.0)$[cite: 2, 4] |
| Top Eigenvector ($\text{PC}_1$) | $\mathbf{v}_1$ | $[0.7071, \; 0.7071]^T$[cite: 2, 4] |

### Step-by-Step Calculation
1. **Standardize each feature[cite: 2, 4]:**
   $$z_1 = \frac{30.0 - 25.0}{11.18} = \frac{5.0}{11.18} \approx 0.4472$$
   $$z_2 = \frac{50.0 - 40.0}{15.0} = \frac{10.0}{15.0} \approx 0.6667$$

2. **Construct matrices[cite: 2, 4]:**
   $$\mathbf{z} = \begin{bmatrix} 0.4472 & 0.6667 \end{bmatrix}_{1 \times 2}, \qquad \mathbf{v}_1 = \begin{bmatrix} 0.7071 \\ 0.7071 \end{bmatrix}_{2 \times 1}$$

3. **Compute dot product ($1\times 2$ multiplied by $2\times 1 \rightarrow 1\times 1$)[cite: 2, 4]:**
   $$y = (0.4472)(0.7071) + (0.6667)(0.7071) = 0.3162 + 0.4714 = \mathbf{0.7876}$$

---

## Execution
```bash
python pca_verify.py
