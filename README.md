# Panel Clustering Estimator (PaCE)

The **Panel Clustering Estimator (PaCE)** estimates heterogeneous treatment effects in panel data
by combining low-rank structure with tree-based clustering of treatment effects.
PaCE is designed for panel settings with repeated observations over units and time, where treatment effects
may be heterogeneous.


## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

## Data format

PaCE expects a long-format pandas DataFrame `df`, where each row corresponds to one (unit, time) pair.

### Required columns

- column_unit: unit identifier (e.g. "unit")
- column_time: time index (e.g. "time")
- column_outcome: outcome variable (e.g. "y")

### Covariates

- columns_for_X: list of covariate column names used for tree splits
  (e.g. ["x1", "x2"])

### Treatments

- columns_for_Z: list of binary treatment indicator column names
  (values should be 0/1 or False/True)


---

## Basic usage

```python
from estimator import TreatmentEffectEstimator

est = TreatmentEffectEstimator(
    data=df,
    column_unit="unit",
    column_time="time",
    column_outcome="y",
    columns_for_X=["x1", "x2"],
    columns_for_Z=["treatA"],
    suggest_r=5,
)

est.fit(max_leaves=10)

print(est.ate)
```

---

## Outputs

After fitting, the estimator provides:

### Average treatment effects

- est.ate  
  A list containing the average treatment effect (ATE) for each treatment in
  columns_for_Z.

### Cluster-level effects

- est.tau  
  Estimated treatment effects for each discovered cluster.

- est.std  
  Corresponding standard errors.





## Example

A complete runnable example using synthetic panel data is provided in:

```
examples/minimal_example.py
```
