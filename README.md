# HW08: PCA and SVD

Name(s):

## Setup

Use Python 3.10 or newer. From this directory:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest -v test_hw08
python3 run_pca.py
python3 run_svd.py
```

## What to implement

Complete these seven TODO functions:

- `PCA.py`: `fit`, `transform`, and `reconstruct_data`.
- `run_pca.py`: `numerical_checks`, `components_for_variance`, and
  `compare_with_sklearn`.
- `run_svd.py`: `low_rank_approximation` (takes the already-computed SVD factors).

Input validation in `validation.py`, figure layouts in `plotting.py`, data
loaders in `data_utils.py`, and the experiment drivers are provided. You do
not need to edit them. The drivers use your PCA methods to produce the figures
and print reconstruction errors. Read them to connect your methods to the
experiments. The SVD driver computes one decomposition and reuses its factors.

The tests cover examples, not every grading requirement. Tests that exercise
unfinished calculations fail until the relevant TODOs are complete. Checks
for the provided validation can pass immediately. Data files are included;
no dataset download is needed at runtime. Read the course HW08 page for all
requirements.

Inputs to PCA are real numeric NumPy arrays; `n_components` is a Python
`int`. You do not need to support lists, strings, complex arrays, or arbitrary
object conversion. Provided validation checks dimensions, finite values, the
component count, nonzero variation, and whether the model has been fitted.

## Written problems

Complete `HW08_written_problems_student.pdf`. The matching `.tex` source is
included if you prefer to typeset your answers. Submit your completed work as
`HW08_written_problems_student.pdf`.

## Part 2: Handwritten digits

### Choosing a smaller representation

**What are the smallest component counts reaching at least 90% and 95% cumulative variance?**

| Cumulative variance | Smallest k |
|---|---:|
| 90% | TODO |
| 95% | TODO |

**Why could multiplying one feature by 1,000 affect PCA?**

**Answer:** _Your answer here._

**When might standardization help?**

**Answer:** _Your answer here._

**How should a zero-variance feature be handled when standardizing?**

**Answer:** _Your answer here._

### Looking at two dimensions

**Which digits overlap in the two-dimensional plot?**

**Answer:** _Your answer here._

**Why does this overlap not establish that the original 64-dimensional data are inseparable?**

**Answer:** _Your answer here._

### Reconstructing the images

**What is the mean squared reconstruction error for each k, over all observations and pixels?**

| k | Mean squared reconstruction error |
|---|---:|
| 2 | TODO |
| 10 | TODO |
| 30 | TODO |
| 64 | TODO |

**How does visual quality change as more components are retained?**

**Answer:** _Your answer here._

**Does retaining 95% variance guarantee 95% classification accuracy? Explain.**

**Answer:** _Your answer here._

### Comparing implementations

**What are the largest absolute differences between your 10-component model and sklearn PCA with svd_solver="full" on the same data?**

| Quantity | Largest absolute difference |
|---|---:|
| Explained-variance ratios | TODO |
| Reconstructed data | TODO |

**Explain the differences. Are any larger than small numerical tolerances?**

**Answer:** _Your answer here._

**If PCA were used before a classifier, which data would you use to fit the mean and directions?**

**Answer:** _Your answer here._

**How would you transform held-out observations?**

**Answer:** _Your answer here._

## Part 3: Low-rank image approximation

**In one short paragraph, compare the row and column meanings in the digits PCA matrix and the single-image matrix, explain why uncentered image approximation is a different task, describe the changes at k = 1, 2, 5, 10, and justify a displayed rank using visual quality and factor-storage counts. Assume equal storage per number; distinguish factor storage from saving a reconstructed image or a PNG.**

**Answer:** _Your answer here._

### Homework questionnaire

1. Approximately how many hours did this homework take?

   **Answer:** TODO

2. How difficult was it from 1 (easy) to 5 (very difficult)?

   **Answer:** TODO

3. What was the biggest challenge you faced?

   **Answer:** TODO

## Submission

Include PCA.py, run_pca.py, run_svd.py, data_utils.py, README.md,
validation.py, plotting.py, requirements.txt, test_hw08.py,
HW08_written_problems_student.pdf, and the four requested
PDF figures in figures/. The grader supplies data/: do not submit data/,
a virtual environment, or installed packages.
# ds_hw8
