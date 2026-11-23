# Task 2: Linear Discriminant Analysis (LDA) Class Separation

This module applies Fisher's Linear Discriminant Analysis (LDA) to find the optimal 1D projection axis $\mathbf{w}$ that maximizes class separability between two labeled groups (`Fail = 0` and `Pass = 1`)[cite: 1].

---

## Files in This Directory
* `lda_analysis.xlsx` — Spreadsheet calculations for class means, scatter matrices, and projections.
* `objective_function.txt` — Complete analytical breakdown of $S_W$, $S_B$, eigen-decomposition, and Fisher criterion evaluation[cite: 1].

---

## Dataset
Two raw features—**Study Hours ($F_1$)** and **Prior Score ($F_2$)**—across 5 samples divided into Class $C_1$ (students 1–2) and Class $C_2$ (students 3–5)[cite: 1]:

| Student ID | Study Hours ($F_1$) | Prior Score ($F_2$) | Class ($y$) |
| :---: | :---: | :---: | :---: |
| 1 | 10 | 50 | $C_1$ (`0 - Fail`) |
| 2 | 20 | 60 | $C_1$ (`0 - Fail`) |
| 3 | 25 | 75 | $C_2$ (`1 - Pass`) |
| 4 | 30 | 85 | $C_2$ (`1 - Pass`) |
| 5 | 40 | 90 | $C_2$ (`1 - Pass`) |

---

## Mathematical Foundations

Unlike PCA (which is unsupervised and maximizes total variance), **LDA is supervised**: it seeks a projection vector $\mathbf{w}$ that simultaneously **pushes class centers far apart** while **keeping samples within each class tightly clustered**[cite: 1].

### 1. Class Mean Vectors ($\mathbf{m}_k$)
For each class $C_k$ with $N_k$ samples:

$$\mathbf{m}_k = \frac{1}{N_k} \sum_{\mathbf{x} \in C_k} \mathbf{x}$$

* **Class 1 ($C_1$):** $\mathbf{m}_1 = \begin{bmatrix} 15.0000 \\ 55.0000 \end{bmatrix}$[cite: 1]
* **Class 2 ($C_2$):** $\mathbf{m}_2 = \begin{bmatrix} 31.6667 \\ 83.3333 \end{bmatrix}$[cite: 1]
* **Mean Difference:** $\mathbf{m}_2 - \mathbf{m}_1 = \begin{bmatrix} 16.6667 \\ 28.3333 \end{bmatrix}$[cite: 1]

---

### 2. Within-Class Scatter Matrix ($S_W$)
Measures how spread out data points are around their own class mean[cite: 1]. Lower scatter means tighter clusters:

$$S_k = \sum_{\mathbf{x} \in C_k} (\mathbf{x} - \mathbf{m}_k)(\mathbf{x} - \mathbf{m}_k)^T, \qquad S_W = S_1 + S_2$$

From the dataset[cite: 1]:

$$S_1 = \begin{bmatrix} 50.0000 & 50.0000 \\ 50.0000 & 50.0000 \end{bmatrix}, \qquad S_2 = \begin{bmatrix} 116.6667 & 108.3333 \\ 108.3333 & 116.6667 \end{bmatrix}$$

$$S_W = S_1 + S_2 = \begin{bmatrix} 166.6667 & 158.3333 \\ 158.3333 & 166.6667 \end{bmatrix}$$

---

### 3. Between-Class Scatter Matrix ($S_B$)
Measures the separation between the centers of the two classes[cite: 1]:

$$S_B = (\mathbf{m}_2 - \mathbf{m}_1)(\mathbf{m}_2 - \mathbf{m}_1)^T = \begin{bmatrix} 277.7778 & 472.2222 \\ 472.2222 & 802.7778 \end{bmatrix}$$

*(Note: Because $S_B$ is an outer product of two vectors, it has rank $1$, yielding at most one non-zero eigenvalue[cite: 1].)*

---

### 4. Fisher's LDA Objective Function ($J(\mathbf{w})$)
LDA maximizes the Rayleigh quotient—the ratio of projected between-class variance to projected within-class variance[cite: 1]:

$$J(\mathbf{w}) = \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$$

Taking the derivative with respect to $\mathbf{w}$ and setting it to zero transforms this optimization into a generalized eigenvalue problem:

$$S_W^{-1} S_B \mathbf{w} = \lambda \mathbf{w}$$

For two classes, the closed-form direction is proportional to:

$$\mathbf{w} \propto S_W^{-1}(\mathbf{m}_2 - \mathbf{m}_1)$$

---

## Computed Results & Interpretation

1. **Eigen-Decomposition of $S_W^{-1} S_B$[cite: 1]:**
   $$S_W^{-1} S_B = \begin{bmatrix} -10.5128 & -17.8718 \\ 12.8205 & 21.7949 \end{bmatrix}$$
   * **Eigenvalues:** $\lambda_1 = \mathbf{11.2821}$, $\lambda_2 = 0.0000$[cite: 1]
   * **Optimal Unit Projection Vector ($\mathbf{w}^*$):**
     $$\mathbf{w}^* = \begin{bmatrix} -0.6341 \\ 0.7733 \end{bmatrix}$$

2. **Objective Ratio Comparison[cite: 1]:**
   | Projection Axis | Vector $\mathbf{w}$ | $J(\mathbf{w})$ Score |
   | :--- | :---: | :---: |
   | Study Hours axis only ($F_1$) | $[1, \; 0]^T$ | $1.6667$ |
   | Prior Score axis only ($F_2$) | $[0, \; 1]^T$ | $4.8167$ |
   | **Optimal LDA Direction ($\mathbf{w}^*$)** | **$[-0.6341, \; 0.7733]^T$** | **$11.2821$** |

   Along $\mathbf{w}^*$, between-class scatter is $\mathbf{w}^T S_B \mathbf{w} = 128.6242$ and within-class scatter is $\mathbf{w}^T S_W \mathbf{w} = 11.4008$, giving $J(\mathbf{w}^*) = \frac{128.6242}{11.4008} = 11.2821$[cite: 1].

3. **Projected Class Means ($\tilde{\mu}_k = \mathbf{w}^{*T} \mathbf{m}_k$)[cite: 1]:**
   * $\tilde{\mu}_1 = 33.0185$
   * $\tilde{\mu}_2 = 44.3598$
   * **1D Class Center Separation:** $|\tilde{\mu}_2 - \tilde{\mu}_1| = \mathbf{11.3413}$

4. **Geometric Insight[cite: 1]:**
   The negative weight on Study Hours ($-0.6341$) paired with the positive weight on Prior Score ($+0.7733$) creates a contrast axis: Prior Score strongly separates Pass from Fail, while subtracting scaled Study Hours cancels out their shared internal correlation within each class, tightening the clusters along the projected line[cite: 1].

