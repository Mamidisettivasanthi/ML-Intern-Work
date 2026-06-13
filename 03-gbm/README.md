# Gradient Boosting Machine (GBM)

> **Fitting the residuals — step by step optimization.**

**What you will learn:** In this guide, you will understand the core concepts of Gradient Boosting Machine (GBM), how to implement it from scratch vs. using Scikit-learn, and how to answer technical interview questions.

---

## 1. What Is Gradient Boosting Machine (GBM)?

Gradient Boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do.

### Real-World Analogy
*Analogy:* Imagine playing golf. Your first shot lands far from the hole. The next shot is taken exactly from where the first landed, aiming only at the remaining distance (the residual). You repeat this until you're in the hole.

---

## 2. Mathematical Formulation

### Negative Gradient (Pseudo-Residuals):

$$ r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)} $$

| Symbol | Meaning |
|---|---|
| $r_{im}$ | Pseudo-residual for sample $i$ at step $m$ |
| $L$ | Differentiable loss function |
| $F_{m-1}(x)$ | Model from previous step |

### Model Update:

$$ F_m(x) = F_{m-1}(x) + \nu \cdot \gamma_m h_m(x) $$

| Symbol | Meaning |
|---|---|
| $F_m(x)$ | Updated model |
| $\nu$ | Learning rate (shrinkage) |
| $h_m(x)$ | New tree fitted to residuals |

---

## 3. How It Works — Step by Step

![Gradient Boosting Machine (GBM) Pipeline](gbm_pipeline.png)

**Step 1:** Initialize the model.
**Step 2:** Iteratively fit to the target (residuals or gradient).
**Step 3:** Optimize the specific loss function using defined parameters.
**Step 4:** Combine outputs into final robust predictions.

---

## 4. Key Assumptions

| Assumption | Why It Matters | What Happens If Violated |
|---|---|---|
| Differentiable Loss | Needs gradients | Algorithm cannot run |
| Weak Learners | Trees must not be too deep | Immediate overfitting |

---

## 5. When to Use / When Not to Use

| ✅ Use When | ❌ Avoid When |
|---|---|
| High accuracy tabular | High dimension sparse data |
| Complex non-linear relations | Low latency requirements |

---

## 6. Implementation Overview

| Aspect | From Scratch (NumPy) | Library (Scikit-learn) |
|---|---|---|
| Target | Negative gradient | `GradientBoostingClassifier` |

### Scikit-learn / Native Library Quick Start

```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
```

---

## 7. Top 5 Interview Questions

**Q1: Why is it called Gradient Boosting?**
- Because it uses Gradient Descent in function space, fitting new models to the negative gradient.

**Q2: What happens if learning rate is 1.0?**
- It takes full steps towards the residuals, which often leads to severe overfitting. Shrinkage is needed.

**Q3: Does GBM run in parallel?**
- No, it is strictly sequential because tree $m$ depends on the output of tree $m-1$.

**Q4: How does it handle classification?**
- By optimizing Log-Loss, and converting raw log-odds scores into probabilities via the sigmoid function.

**Q5: What is the difference between AdaBoost and GBM?**
- AdaBoost reweights data points based on errors; GBM fits new trees directly to the residual errors.

---

## 8. Quick Reference Table

| Item | Detail |
|------|--------|
| **Algorithm Type** | Ensemble Learning |
| **Strengths** | Extremely high accuracy |
| **Weaknesses** | Can be complex to tune |

---

## 9. References & Further Reading

| Resource | Link |
|---|---|
| Paper | [Greedy Function Approximation](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) |

---

## 10. Environment & Setup

To run the accompanying Jupyter Notebook, ensure you have the following installed:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```
For specific libraries, see the top cell of the Jupyter Notebook.
