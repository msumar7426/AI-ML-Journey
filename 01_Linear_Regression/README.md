# Simple Linear Regression

## Purpose

This document is the long-term reference for the Simple Linear Regression learning block
in the AI-ML-Journey repository.

It is written for a beginner who may have no previous Machine Learning experience.

The goal is not only to remember code. The goal is to understand:

- what Linear Regression is;
- why we use it;
- how the data is represented;
- how Pandas objects behave;
- how training and testing work;
- what `fit()` and `predict()` do;
- what slope and intercept mean;
- how prediction errors are calculated;
- what MAE, MSE and RMSE measure;
- what R² means;
- why R² can be negative;
- what Adjusted R² means;
- how manual calculations relate to scikit-learn;
- common mistakes;
- interview questions;
- practice problems;
- the complete workflow used in the notebook.

The accompanying notebook is `learning.ipynb`.

The dataset used for the learning example is `placement.csv`.

---

## Table of Contents

1. [Learning Goal](#1-learning-goal)
2. [What Is Machine Learning?](#2-what-is-machine-learning)
3. [What Is Linear Regression?](#3-what-is-linear-regression)
4. [Simple Linear Regression](#4-simple-linear-regression)
5. [The CGPA to Package Problem](#5-the-cgpa-to-package-problem)
6. [Important Vocabulary](#6-important-vocabulary)
7. [Mathematical Notation](#7-mathematical-notation)
8. [The Dataset](#8-the-dataset)
9. [Loading the Dataset](#9-loading-the-dataset)
10. [Pandas Series and DataFrame](#10-pandas-series-and-dataframe)
11. [Important Pandas Operations](#11-important-pandas-operations)
12. [Understanding the Dataset](#12-understanding-the-dataset)
13. [Descriptive Statistics](#13-descriptive-statistics)
14. [Mean, Median and Dispersion](#14-mean-median-and-dispersion)
15. [Standard Deviation](#15-standard-deviation)
16. [Visualization](#16-visualization)
17. [Features and Target](#17-features-and-target)
18. [Why `df['cgpa']` and `df[['cgpa']]` Differ](#18-why-dfcgpa-and-dfcgpa-differ)
19. [Train and Test Split](#19-train-and-test-split)
20. [X_train, X_test, y_train and y_test](#20-x_train-x_test-y_train-and-y_test)
21. [Training the Model](#21-training-the-model)
22. [What Training Means](#22-what-training-means)
23. [The Linear Regression Equation](#23-the-linear-regression-equation)
24. [Slope](#24-slope)
25. [Intercept](#25-intercept)
26. [Prediction](#26-prediction)
27. [Actual vs Predicted Values](#27-actual-vs-predicted-values)
28. [Comparison Table](#28-comparison-table)
29. [Residuals and Errors](#29-residuals-and-errors)
30. [Mean Absolute Error](#30-mean-absolute-error)
31. [Mean Squared Error](#31-mean-squared-error)
32. [Root Mean Squared Error](#32-root-mean-squared-error)
33. [MAE vs MSE vs RMSE](#33-mae-vs-mse-vs-rmse)
34. [Manual Metrics vs scikit-learn](#34-manual-metrics-vs-scikit-learn)
35. [Floating-Point Precision](#35-floating-point-precision)
36. [R² Score](#36-r2-score)
37. [The Mean Baseline](#37-the-mean-baseline)
38. [SST](#38-sst)
39. [SSE](#39-sse)
40. [Understanding the R² Formula](#40-understanding-the-r2-formula)
41. [R² Equal to 1, 0 and Negative](#41-r2-equal-to-1-0-and-negative)
42. [R² on the CGPA Dataset](#42-r2-on-the-cgpa-dataset)
43. [Adjusted R²](#43-adjusted-r2)
44. [n and k](#44-n-and-k)
45. [Adjusted R² Example](#45-adjusted-r2-example)
46. [R² vs Adjusted R²](#46-r2-vs-adjusted-r2)
47. [Complete Notebook Workflow](#47-complete-notebook-workflow)
48. [Important Code Snippets](#48-important-code-snippets)
49. [Common Mistakes](#49-common-mistakes)
50. [Mathematics Revision Sheet](#50-mathematics-revision-sheet)
51. [Interview Questions](#51-interview-questions)
52. [Practice Questions](#52-practice-questions)
53. [Six-Month Revision Checklist](#53-six-month-revision-checklist)
54. [Learning Results](#54-learning-results)
55. [Next Step](#55-next-step)
56. [Final Mental Model](#56-final-mental-model)

---

# 1. Learning Goal

The learning example uses a small placement dataset containing `CGPA` and `Package`.

The task is:

```text
CGPA
  |
  v
Linear Regression
  |
  v
Predicted Package
```

This is a teaching example. It is useful because the dataset is small enough to
understand manually.
The real portfolio project should use a different and more realistic dataset.

---

# 2. What Is Machine Learning?

Machine Learning is a way of building systems that learn patterns from data and use
those patterns to make predictions or decisions.

A traditional program can be represented as:

```text
Rules + Data
     |
     v
   Output
```

A supervised Machine Learning workflow can be represented as:

```text
Input Data + Known Outputs
          |
          v
       Learning
          |
          v
        Model
          |
          v
New Input -> Prediction
```

For this learning example:

```text
Student examples
      |
      v
CGPA + Package
      |
      v
Linear Regression
      |
      v
Learn relationship
      |
      v
New CGPA
      |
      v
Predicted Package
```

The model does not understand the problem like a human. It receives numerical data and
learns parameters that allow it to produce predictions.

---

# 3. What Is Linear Regression?

Linear Regression is a supervised Machine Learning algorithm used to model a
relationship between one or more input variables and a numerical target.

The central idea is to find a linear relationship that describes the data as well as
possible.

For one input variable:

```text
One feature -> One numerical target
```

Example:

```text
CGPA -> Package
```

A simple linear model has the form:

> **ŷ = mx + b**

Here:

- `x` is the input feature;
- `y` is the actual target;
- `ŷ` is the predicted target;
- `m` is the slope;
- `b` is the intercept.

The hat accent over `y` (`ŷ`) indicates a predicted or estimated value, distinguishing it from the actual observed target value `y`.

---

# 4. Simple Linear Regression

"Simple" means that the model uses one input feature.

For this example:

```text
X = CGPA
y = Package
```

The model is:

> **ŷ = mx + b**

For our problem:

> **ŷ = m(CGPA) + b**

In plain English:

> Predicted package = slope times CGPA plus intercept.

The model learns suitable values for `m` and `b` from the training data.

## What does the hat mean?

The notation `ŷ` (read as "y-hat") represents an estimated or predicted value. The hat accent (`ŷ`) tells us that the value is an estimate generated by the model rather than the true observed target `y`.

Similarly, `ȳ` (read as "y-bar") represents the arithmetic mean of the target values `y`.

---

# 5. The CGPA to Package Problem

The dataset contains:

```text
cgpa
package
```

We want to learn whether CGPA can be used to predict package.

The relationship can be represented as:

```text
CGPA -> Package
```

The relationship is not expected to be perfect.

A student with a higher CGPA may receive a higher package, but package can also depend
on many other factors.

Examples include:

- skills;
- university;
- internships;
- interview performance;
- communication skills;
- job market conditions;
- company;
- location;
- experience.

For the learning example, we intentionally use only CGPA.

This makes the mathematics easy to understand.

---

# 6. Important Vocabulary

## Feature

A feature is an input variable used by the model.

Here:

```text
CGPA
```

is the feature.

## Target

The target is the value we want the model to predict.

Here:

```text
Package
```

is the target.

## Observation

An observation is one row of data.

For example:

```text
CGPA = 7.82
Package = 3.7
```

represents one observation.

## Model

A model is the learned relationship used to make predictions.

For Simple Linear Regression:

> **ŷ = mx + b**

## Training

Training means using training data to learn the model parameters.

## Testing

Testing means evaluating the trained model on data that was not used to fit the model.

## Prediction

A prediction is the output generated by the trained model for an input.

## Residual

A residual is the difference between an actual value and its prediction.

> **eᵢ = yᵢ - ŷᵢ**

---

# 7. Mathematical Notation

| Symbol | Meaning |
|---|---|
| `x` | Input feature |
| `y` | Actual target |
| `ŷ` | Predicted target |
| `m` | Slope |
| `b` | Intercept |
| `eᵢ` | Error or residual for observation `i` |
| `ȳ` | Mean of the actual target values |
| `n` | Number of observations |
| `k` | Number of predictors/features |
| `Σ` | Summation |
| `R²` | Coefficient of determination |
| `SST` | Total Sum of Squares |
| `SSE` | Sum of Squared Errors |

---

# 8. The Dataset

The learning dataset contains:

```text
200 rows
2 columns
```

The columns are:

```text
cgpa
package
```

The dataset inspected during the learning process contained:

```text
cgpa      -> 200 non-null values
package   -> 200 non-null values
```

Both columns were numerical.

The dataset is used to demonstrate the complete basic regression workflow.

---

# 9. Loading the Dataset

The notebook and dataset are now in the same directory:

```text
01_Linear_Regression/
    learning.ipynb
    placement.csv
    README.md
```

Therefore, the notebook can load the dataset with:

```python
import pandas as pd

df = pd.read_csv("placement.csv")
```

## What does `pd.read_csv()` do?

```python
pd.read_csv("placement.csv")
```

means:

> Use Pandas to read the CSV file and create a DataFrame.

We store the result in:

```python
df
```

`df` is only a variable name. It is commonly used as shorthand for DataFrame, but Python
does not require that name.

This would also work:

```python
data = pd.read_csv("placement.csv")
```

---

# 10. Pandas Series and DataFrame

Understanding Series and DataFrame is important because Machine Learning libraries
expect data in specific shapes.

## Series

Selecting one column with one pair of brackets:

```python
df['cgpa']
```

usually returns a Pandas Series.

Think:

```text
Series
  |
  v
One column
  |
  v
One-dimensional
```

For 200 observations:

```text
(200,)
```

## DataFrame

Selecting a column with two pairs of brackets:

```python
df[['cgpa']]
```

returns a one-column DataFrame.

Think:

```text
DataFrame
  |
  v
Table
  |
  v
Two-dimensional
```

Its shape is:

```text
(200, 1)
```

Even though it contains only one column, it is still a two-dimensional table.

---

# 11. Important Pandas Operations

## `shape`

```python
df.shape
```

returns:

```text
(rows, columns)
```

For our dataset:

```text
(200, 2)
```

Therefore:

```python
df.shape[0]
```

returns `200` and:

```python
df.shape[1]
```

returns `2`.

Memory trick:

```text
shape[0] -> rows
shape[1] -> columns
```

## `info()`

```python
df.info()
```

provides structural information such as:

- number of rows;
- column names;
- non-null counts;
- data types;
- memory usage.

## `describe()`

```python
df.describe()
```

provides descriptive statistics for numerical columns.

## Select one column

```python
df['cgpa']
```

Returns a Series.

## Select one column as a DataFrame

```python
df[['cgpa']]
```

Returns a DataFrame.

## Select multiple columns

```python
df[['cgpa', 'package']]
```

Returns a DataFrame.

## `.iloc`

`.iloc` performs integer-location based indexing.

For example:

```python
df.iloc[0]
```

means:

> Select the row at integer position 0.

This:

```python
df.iloc[0:5]
```

selects positions 0 through 4.

This:

```python
df.iloc[:, 0]
```

means all rows from the first column.

The general form is:

```python
df.iloc[row_position, column_position]
```

---

# 12. Understanding the Dataset

## `shape`

```python
df.shape
```

answers:

> How many rows and columns are present?

## `info()`

```python
df.info()
```

answers questions such as:

- What are the columns?
- How many non-null values are present?
- What are the data types?
- How many rows exist?

## `describe()`

```python
df.describe()
```

helps us understand the numerical distribution.

The learning dataset showed approximately:

```text
CGPA
mean  ≈ 6.99
std   ≈ 1.07
min   = 4.26
max   = 9.58

Package
mean  ≈ 3.00
std   ≈ 0.69
min   = 1.37
max   = 4.62
```

These values describe this dataset. They do not prove that CGPA causes package to
increase.

---

# 13. Descriptive Statistics

## Count

Count tells us how many non-missing observations are present.

## Mean

The arithmetic mean is:

> **\bar{x} = (Σ  xᵢ / n)**

For:

```text
5, 5, 5, 5, 5
```

the mean is:

> **\bar{x} = (25 / 5) = 5**

## Median

The median is the middle value after sorting.

For:

```text
1, 2, 3, 4, 5
```

the median is `3`.

For:

```text
1, 2, 3, 4
```

the median is:

> **(2+3 / 2) = 2.5**

## Minimum

The smallest value.

## Maximum

The largest value.

## Quartiles

```text
25% -> Q1
50% -> Q2, median
75% -> Q3
```

---

# 14. Mean, Median and Dispersion

Consider:

```text
5, 5, 5, 5, 5
```

The mean is 5.

The median is 5.

The values have almost no spread.

Now consider:

```text
1, 3, 5, 7, 9
```

The mean is still 5.

The median is still 5.

However, the values are much more spread out.

This gives an important lesson:

> Mean and median describe the center of the data, but they do not completely describe
> how spread out the data is.

Measures such as standard deviation describe dispersion.

---

# 15. Standard Deviation

Standard deviation is a measure of how spread out observations are around their mean.

A larger standard deviation generally indicates greater spread.

For the learning dataset, the approximate values were:

```text
CGPA standard deviation     ≈ 1.069
Package standard deviation  ≈ 0.692
```

Do not interpret this as:

> Every CGPA value is exactly 1.069 away from the mean.

That is incorrect.

Standard deviation describes overall dispersion. It does not say that every observation
is exactly one standard deviation away from the mean.

---

# 16. Visualization

We used Matplotlib:

```python
import matplotlib.pyplot as plt
```

The scatter plot was:

```python
plt.scatter(df['cgpa'], df['package'])

plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('CGPA vs Package')

plt.show()
```

## What does each command do?

```python
plt.xlabel('CGPA')
```

names the x-axis.

```python
plt.ylabel('Package')
```

names the y-axis.

```python
plt.title('CGPA vs Package')
```

gives the plot a title.

```python
plt.scatter(df['cgpa'], df['package'])
```

creates a scatter plot using CGPA on the x-axis and package on the y-axis.

```python
plt.show()
```

displays the plot.

---

# 17. Features and Target

For our model:

```python
X = df[['cgpa']]
y = df['package']
```

## X

`X` contains the input feature.

Here:

```text
CGPA
```

## y

`y` contains the target.

Here:

```text
Package
```

Think:

```text
X
 |
 v
Input
 |
 v
Model
 |
 v
Prediction
```

The actual target values are represented by `y`.

---

# 18. Why `df['cgpa']` and `df[['cgpa']]` Differ

This is one of the most important Pandas concepts from the learning process.

## One pair of brackets

```python
df['cgpa']
```

returns a Series.

Its shape is:

```text
(200,)
```

This means 200 values in one dimension.

## Two pairs of brackets

```python
df[['cgpa']]
```

returns a DataFrame.

Its shape is:

```text
(200, 1)
```

This means 200 rows and one feature in two dimensions.

## Why does this matter?

Machine Learning libraries commonly represent features as a matrix with the shape:

```text
(number_of_samples, number_of_features)
```

Therefore:

```text
200 observations
1 feature
```

becomes:

```text
(200, 1)
```

This is why we commonly write:

```python
X = df[['cgpa']]
```

For the target, a one-dimensional Series is commonly suitable:

```python
y = df['package']
```

with shape:

```text
(200,)
```

## Visual memory

```text
df['cgpa']
    |
    v
  Series
    |
    v
   1D
    |
    v
 (200,)


df[['cgpa']]
    |
    v
 DataFrame
    |
    v
   2D
    |
    v
 (200, 1)
```

---

# 19. Train and Test Split

We used:

```python
from sklearn.model_selection import train_test_split
```

Then:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

With 200 observations and `test_size=0.2`:

```text
80% -> training
20% -> testing
```

So:

```text
Training -> 160 observations
Testing  -> 40 observations
```

## Why split the data?

The model should be evaluated on observations it did not use during training.

The basic workflow is:

```text
Training data
      |
      v
Learn model
      |
      v
Test data
      |
      v
Make predictions
      |
      v
Evaluate
```

The purpose is to estimate how well the model generalizes to unseen data.

Do not use this oversimplification:

> We split the data because otherwise the model memorizes everything.

The better explanation is:

> We separate training and testing data so that model performance can be evaluated on
> observations that were not used to fit the model.

---

# 20. X_train, X_test, y_train and y_test

| Variable | Meaning |
|---|---|
| `X_train` | Input features used for training |
| `y_train` | Actual targets corresponding to training inputs |
| `X_test` | Input features held out for testing |
| `y_test` | Actual targets corresponding to test inputs |
| `y_pred` | Predictions produced by the trained model |

The most important distinction is:

```text
y_test
  |
  v
Actual target values
```

while:

```text
y_pred
  |
  v
Model predictions
```

`y_test` is not the prediction.

---

# 21. Training the Model

Import Linear Regression:

```python
from sklearn.linear_model import LinearRegression
```

Create the model:

```python
model = LinearRegression()
```

Train it:

```python
model.fit(X_train, y_train)
```

The important idea is that `model.fit()` learns from the training data.

---

# 22. What Training Means

When we write:

```python
model.fit(X_train, y_train)
```

we are asking the algorithm to learn the relationship between the training features and
training targets.

For Simple Linear Regression, the model learns:

```text
m -> slope
b -> intercept
```

The resulting model has the form:

> **ŷ = mx + b**

Training should not be thought of as human-style memorization.

The algorithm estimates parameters according to its optimization procedure.

For ordinary least squares Linear Regression, the fitted line is chosen to minimize the
sum of squared residuals on the training data.

---

# 23. The Linear Regression Equation

The basic equation is:

> **ŷ = mx + b**

Where:

- `x` is the input feature;
- `ŷ` is the predicted output;
- `m` is the slope;
- `b` is the intercept.

For our problem:

> **ŷ = m(CGPA) + b**

In plain English:

> Predicted package equals slope times CGPA plus intercept.

The model learns `m` and `b`.

---

# 24. Slope

The slope tells us how much the predicted output changes when the input increases by one
unit.

For example, if:

> **m = 0.5**

then increasing CGPA by one point changes the predicted package by 0.5 package units
according to the fitted line.

The slope can be:

```text
positive
negative
zero
```

A positive slope means the predicted target increases as the feature increases.

A negative slope means the predicted target decreases as the feature increases.

A zero slope means the fitted line is horizontal.

Important:

> The slope describes the relationship represented by the fitted model. It does not by
> itself prove causation.

---

# 25. Intercept

The intercept is the predicted target when the input is zero.

Set:

> **x=0**

in:

> **ŷ=mx+b**

Then:

> **ŷ=m(0)+b**

Therefore:

> **ŷ=b**

The intercept does not always have a useful real-world interpretation.

If CGPA = 0 is outside the meaningful range of the dataset, interpreting the intercept
as a real-world package at CGPA 0 may not be meaningful.

---

# 26. Prediction

After training:

```python
y_pred = model.predict(X_test)
```

This means:

> Give the trained model the test input values and ask it to produce predictions.

For example, `y_pred` might contain:

```text
[2.7803, 3.1363, 3.1995, ...]
```

Each prediction corresponds to one row of `X_test`.

The workflow is:

```text
X_test
  |
  v
Model
  |
  v
y_pred
```

while actual values are:

```text
X_test
  |
  v
Actual values
  |
  v
y_test
```

---

# 27. Actual vs Predicted Values

We have:

```text
y_test
  |
  v
Actual package
```

and:

```text
y_pred
  |
  v
Predicted package
```

Example:

```text
Actual     Predicted
3.00       2.78
3.20       3.14
3.50       3.20
```

The difference between actual and predicted values is the residual.

---

# 28. Comparison Table

We created a DataFrame to compare actual and predicted values:

```python
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print(comparison)
```

## `pd.DataFrame(...)`

Creates a Pandas DataFrame and stores it in `comparison`.

## `y_test.values`

Creates the `Actual` column from the actual test target values.

`y_test` is a Pandas Series. `.values` extracts its underlying array-like values.

## `y_pred`

Creates the `Predicted` column.

`y_pred` returned by `model.predict()` is already a NumPy array, so `y_pred.values` is
not needed.

In general, `y_pred.values` would fail because NumPy arrays do not have the same
`.values` attribute used by Pandas objects.

---

# 29. Residuals and Errors

We created:

```python
comparison['Error'] = (
    comparison['Actual'] - comparison['Predicted']
)
```

This creates a new column called `Error`.

For one observation:

```text
Actual    = 3.5
Predicted = 3.2
```

then:

```text
Error = 3.5 - 3.2
      = 0.3
```

If:

```text
Actual    = 3.2
Predicted = 3.5
```

then:

```text
Error = 3.2 - 3.5
      = -0.3
```

The sign tells us the direction of the residual.

```text
Positive -> actual > predicted
Negative -> actual < predicted
Zero     -> exact prediction
```

The mathematical definition is:

> **eᵢ = yᵢ - ŷᵢ**

---

# 30. Mean Absolute Error

MAE means Mean Absolute Error.

The formula is:

> **MAE = (1/n) · Σ |yᵢ-ŷᵢ|**

In simple words:

1. Calculate each error.
2. Take the absolute value.
3. Add the absolute errors.
4. Divide by the number of observations.

## Example

Suppose the errors are:

```text
2, -5, 3, 4
```

Absolute errors:

```text
2, 5, 3, 4
```

Sum:

> **2+5+3+4=14**

MAE:

> **MAE=(14 / 4)=3.5**

## Why absolute value?

Without absolute value, positive and negative errors could cancel.

For example:

```text
-5 + 5 = 0
```

That would incorrectly suggest zero average error.

MAE prevents this cancellation.

## Interpretation

If:

```text
MAE = 0.23
```

the average absolute prediction error is about 0.23 target units.

MAE is easy to interpret because it is expressed in the original target units.

## scikit-learn

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(mae)
```

---

# 31. Mean Squared Error

MSE means Mean Squared Error.

The formula is:

> **MSE=(1/n) · Σ (yᵢ-ŷᵢ)²**

Steps:

1. Calculate the errors.
2. Square the errors.
3. Add the squared errors.
4. Divide by the number of observations.

## Example

Errors:

```text
2, -5, 3, 4
```

Squared errors:

```text
4, 25, 9, 16
```

Sum:

> **4+25+9+16=54**

MSE:

> **MSE=(54 / 4)=13.5**

## Why square the errors?

Squaring does two important things.

### 1. It removes the sign

Both `-5` and `5` become `25` after squaring.

### 2. It gives larger errors more influence

For example:

> **2²=4**

while:

> **5²=25**

Therefore, MSE is more sensitive to large errors than MAE.

## scikit-learn

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(mse)
```

---

# 32. Root Mean Squared Error

RMSE means Root Mean Squared Error.

The formula is:

> **RMSE=√{MSE}**

For the previous example:

> **MSE=13.5**

Therefore:

> **RMSE=√{13.5}**

> **RMSE\approx3.6742**

## Why take the square root?

MSE is expressed in squared target units.

If package is measured in package units, MSE is in package-squared units.

Taking the square root brings the metric back to the original target units.

This makes RMSE easier to interpret.

## Python

```python
rmse = np.sqrt(mse)
print(rmse)
```

---

# 33. MAE vs MSE vs RMSE

| Metric | Main operation | Large errors | Original units |
|---|---|---|---|
| MAE | Absolute value | Less sensitive | Yes |
| MSE | Squaring | More sensitive | No |
| RMSE | Square root of MSE | More sensitive | Yes |

Memory:

```text
MAE
 |
 v
Absolute

MSE
 |
 v
Square

RMSE
 |
 v
Square + Root
```

There is no universally best metric. The appropriate metric depends on the problem and
what types of errors matter.

---

# 34. Manual Metrics vs scikit-learn

We manually calculated the metrics to understand their mathematics.

We then compared the results with scikit-learn.

For example, manual MAE:

```python
mae = comparison['Absolute Error'].mean()
```

scikit-learn MAE:

```python
from sklearn.metrics import mean_absolute_error

mae_sklearn = mean_absolute_error(y_test, y_pred)
```

The difference was:

```text
0.0
```

This verifies that our manual calculation matches the library calculation.

## Why use scikit-learn in real projects?

Because it is:

- standard;
- tested;
- concise;
- less error-prone than repeatedly writing metric formulas manually.

## Why learn the manual calculation?

Because understanding the mathematics helps us:

- interpret metrics;
- debug code;
- answer interview questions;
- choose appropriate metrics;
- avoid treating libraries as black boxes.

The goal is not to manually reimplement every library function in production.
The goal is to understand what the function is doing.

---

# 35. Floating-Point Precision

When manual R² and scikit-learn R² were compared, the results were:

```text
Manual R²:
0.7730984312051674

scikit-learn R²:
0.7730984312051673
```

The difference was:

```text
1.1102230246251565e-16
```

This is effectively zero for this comparison.

Computers represent many decimal values using finite binary floating-point
representations.
Small numerical differences can therefore appear.

Use:

```python
np.isclose(r2_manual, r2_sklearn)
```

to test whether two floating-point results are sufficiently close.

It should return:

```text
True
```

Do not interpret a difference around `1e-16` as meaningful model error.

---

# 36. R² Score

R² is called the coefficient of determination.

The formula is:

> **R²=1-(SSE / SST)**

R² compares the model against a simple baseline that predicts the mean of the target.

The key intuition is:

> How much better is the model than simply predicting the mean?

Another useful interpretation is:

> R² measures the proportion of variation in the target explained by the model relative
> to the mean baseline.

R² is not ordinary classification accuracy.

---

# 37. The Mean Baseline

Suppose the actual values are:

```text
10, 20, 30, 40
```

The mean is:

> **ȳ=(10+20+30+40 / 4)=25**

A simple baseline can predict the mean for every observation:

```text
25
25
25
25
```

This baseline is important because R² asks how the model performs relative to this
strategy.

---

# 38. SST

SST means Total Sum of Squares.

The formula is:

> **SST=Σ (yᵢ-ȳ)²**

It measures the total squared variation of the actual target values around their mean.

For:

```text
Actual = [10, 20, 30, 40]
Mean   = 25
```

we calculate:

> **(10-25)²+(20-25)²+(30-25)²+(40-25)²**

> **=225+25+25+225**

Therefore:

> **SST=500**

Python:

```python
sst = sum((y - mean) ** 2 for y in actual)
```

Read this as:

> For every actual value, subtract the mean, square the result, and add everything
> together.

---

# 39. SSE

SSE means Sum of Squared Errors.

The formula is:

> **SSE=Σ (yᵢ-ŷᵢ)²**

It measures the total squared prediction error made by the model.

Python:

```python
sse = sum(
    (y - y_pred_value) ** 2
    for y, y_pred_value in zip(actual, predicted)
)
```

Read this as:

> Take an actual value and its prediction, calculate the error, square it, and add the
> result for every observation.

The key distinction is:

```text
SST -> squared error of the mean baseline
SSE -> squared error of our model
```

---

# 40. Understanding the R² Formula

The formula is:

> **R²=1-(SSE / SST)**

Remember:

```text
SST -> baseline error
SSE -> model error
```

So R² asks how much of the baseline error the model eliminates.

## If the model is perfect

If:

> **SSE=0**

then:

> **R²=1**

## If the model equals the mean baseline

If:

> **SSE=SST**

then:

> **R²=0**

## If the model is worse than the baseline

If:

> **SSE>SST**

then:

> **R²<0**

This is why R² can be negative.

---

# 41. R² Equal to 1, 0 and Negative

## R² = 1

Suppose:

```text
Actual:
10, 20, 30, 40

Predicted:
10, 20, 30, 40
```

Every prediction is exact.

Therefore:

> **SSE=0**

and:

> **R²=1**

Interpretation:

> Perfect predictions for the evaluated observations.

## R² = 0

Suppose the mean is 25 and the model predicts:

```text
25, 25, 25, 25
```

This is exactly the mean baseline.

Therefore:

> **SSE=SST**

and:

> **R²=0**

Interpretation:

> The model performs no better than the mean baseline under the R² definition.

## Negative R²

Suppose:

```text
Actual:
10, 20, 30, 40

Predicted:
50, 50, 50, 50
```

We get:

> **SSE=3000**

and:

> **SST=500**

Therefore:

> **R²=1-(3000 / 500)**

> **R²=-5**

Interpretation:

> The model performs worse than the mean baseline.

## Important

Do not say:

```text
R² = 0.773
```

means:

```text
77.3% accuracy
```

That is not a correct interpretation.

---

# 42. R² on the CGPA Dataset

Our CGPA to Package model produced approximately:

```text
Manual R²:
0.7730984312051674

scikit-learn R²:
0.7730984312051673
```

The difference was:

```text
1.1102230246251565e-16
```

These values are effectively identical.

Rounded:

> **R²\approx0.7731**

Interpretation:

> The model explains approximately 77.3% of the variation in package values relative to
> the mean baseline on this evaluation set.

This does not mean:

> The model is 77.3% accurate.

---

# 43. Adjusted R²

Adjusted R² modifies R² by accounting for the number of predictors in the model.

The formula is:

> **R²_{\mathrm{adj}}
=
1-
((1-R²)(n-1) / n-1-k)**

The intuition is:

```text
R²
 |
 v
Explained variation

Adjusted R²
 |
 v
Explained variation
+
model complexity
```

Ordinary R² does not decrease simply because predictors are added.

Adjusted R² introduces a complexity penalty.

This makes it useful when comparing models with different numbers of predictors.

---

# 44. n and k

These symbols are important.

## n

`n` is the number of observations used in the calculation.

For example:

```text
5 rows
```

means:

> **n=5**

## k

`k` is the number of predictors or input features.

For:

```text
CGPA + Age -> Package
```

there are two predictors:

> **k=2**

For:

```text
CGPA -> Package
```

there is one predictor:

> **k=1**

## Our evaluation

Our test set contained 40 observations.

Therefore:

> **n=40**

Our model had one predictor, CGPA.

Therefore:

> **k=1**

---

# 45. Adjusted R² Example

Suppose:

> **R²=0.80,\quad n=10,\quad k=2**

The formula is:

> **R²_{\mathrm{adj}}
=
1-
((1-0.80)(10-1) / 10-1-2)**

Simplify:

> **=1-(0.20 × 9 / 7)**

> **=1-(1.8 / 7)**

Therefore:

> **R²_{\mathrm{adj}}\approx0.742857**

Now keep:

> **R²=0.80,\quad n=10**

but increase the number of predictors to:

> **k=3**

Then:

> **R²_{\mathrm{adj}}=0.70**

This demonstrates the complexity penalty.

## Important nuance

Do not say:

> Adjusted R² always decreases when a feature is added.

The correct statement is:

> Adjusted R² can decrease when a new feature does not improve the model enough to
> justify the added complexity.

It can also increase when the added feature provides enough improvement.

---

# 46. R² vs Adjusted R²

| R² | Adjusted R² |
|---|---|
| Explained variation relative to mean | Explained variation plus predictor count |
| Does not fall from added predictors | Can fall if added predictors add little value |
| Useful for evaluation | Useful for model comparison |
| Simpler metric | Includes a complexity adjustment |

Memory:

```text
R²
 |
 v
How much variation is explained?

Adjusted R²
 |
 v
How much variation is explained after accounting
for the number of predictors?
```

---

# 47. Complete Notebook Workflow

The learning notebook follows this workflow:

```text
Load libraries
      |
      v
Load dataset
      |
      v
Explore dataset
      |
      +--> shape
      |
      +--> info()
      |
      +--> describe()
      |
      v
Understand statistics
      |
      v
Visualize
      |
      v
Define X and y
      |
      v
Train/test split
      |
      v
Create Linear Regression model
      |
      v
Fit model
      |
      v
Predict
      |
      v
Compare actual vs predicted
      |
      v
Calculate errors
      |
      +--> MAE
      |
      +--> MSE
      |
      +--> RMSE
      |
      +--> R²
      |
      +--> Adjusted R²
      |
      v
Interpret results
```

We deliberately used tiny mathematical examples for R² and Adjusted R².

The purpose was:

```text
Tiny dataset
     |
     v
Understand mathematics

Real dataset
     |
     v
Apply ML workflow
```

---

# 48. Important Code Snippets

## Import libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Load CSV

```python
df = pd.read_csv("placement.csv")
```

## Dimensions

```python
df.shape
df.shape[0]  # rows
df.shape[1]  # columns
```

## Information

```python
df.info()
```

## Statistics

```python
df.describe()
```

## Select one column

```python
df['cgpa']
```

## Select a column as a DataFrame

```python
df[['cgpa']]
```

## Features and target

```python
X = df[['cgpa']]
y = df['package']
```

## Train/test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

## Create model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

## Train

```python
model.fit(X_train, y_train)
```

## Predict

```python
y_pred = model.predict(X_test)
```

## Inspect learned parameters

```python
print(model.coef_)
print(model.intercept_)
```

For Simple Linear Regression:

```text
model.coef_[0] -> slope
model.intercept_ -> intercept
```

## Comparison

```python
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
```

## Error

```python
comparison['Error'] = (
    comparison['Actual'] - comparison['Predicted']
)
```

## Absolute Error

```python
comparison['Absolute Error'] = (
    comparison['Error'].abs()
)
```

## Squared Error

```python
comparison['Squared Error'] = (
    comparison['Error'] ** 2
)
```

## MAE

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
```

## MSE

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
```

## RMSE

```python
rmse = np.sqrt(mse)
```

## R²

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
```

## Adjusted R²

```python
n = len(y_test)
k = X_test.shape[1]

adjusted_r2 = 1 - (
    ((1 - r2) * (n - 1))
    / (n - 1 - k)
)
```

---

# 49. Common Mistakes

## Mistake 1: Confusing Series and DataFrame

```python
df['cgpa']
```

usually returns a Series.

```python
df[['cgpa']]
```

returns a DataFrame.

Remember:

```text
(200,)    -> one-dimensional
(200, 1)  -> two-dimensional
```

## Mistake 2: Thinking `y_test` is the prediction

Wrong:

```text
y_test = prediction

```

Correct:

```text
y_test -> actual unseen target values
y_pred -> model predictions
```

## Mistake 3: Calling R² accuracy

Wrong:

> R² = 77% accuracy.

Better:

> R² = 0.77 means the model explains approximately 77% of the variation relative to the
> mean baseline for that evaluation.

## Mistake 4: Thinking one zero error means the whole model has zero error

If one observation has:

```text
Error = 0
```

that prediction is perfect.

It does not mean the entire model has zero error.

## Mistake 5: Forgetting why MSE squares errors

Squaring prevents positive and negative errors from cancelling and gives larger errors
more influence.

## Mistake 6: Thinking RMSE is unrelated to MSE

RMSE is simply:

> **RMSE=√{MSE}**

## Mistake 7: Thinking negative R² is impossible

Negative R² is possible.

It means the model performs worse than the mean baseline under the R² definition.

## Mistake 8: Misinterpreting standard deviation

Standard deviation measures overall spread. It does not mean every observation is
exactly that distance from the mean.

## Mistake 9: Confusing n and k

```text
n -> number of observations
k -> number of predictors
```

## Mistake 10: Evaluating only training data

Training performance alone does not tell us how well the model generalizes to unseen
observations.

## Mistake 11: Treating correlation as causation

A positive relationship between CGPA and package does not prove that increasing CGPA
causes package to increase.

## Mistake 12: Assuming a high R² automatically means a good model

A model can have a high R² and still be unsuitable for a particular business problem.

Evaluation must consider:

- the target;
- the data;
- the test setup;
- the metric;
- the practical use case;
- assumptions and limitations.

---

# 50. Mathematics Revision Sheet

## Mean

> **\bar{x}=(Σ  xᵢ / n)**

## Linear equation

> **y=mx+b**

## Prediction

> **ŷ=mx+b**

## Residual

> **eᵢ=yᵢ-ŷᵢ**

## MAE

> **MAE=(1/n) · Σ |yᵢ-ŷᵢ|**

## MSE

> **MSE=(1/n) · Σ (yᵢ-ŷᵢ)²**

## RMSE

> **RMSE=√{MSE}**

## SST

> **SST=Σ (yᵢ-ȳ)²**

## SSE

> **SSE=Σ (yᵢ-ŷᵢ)²**

## R²

> **R²=1-(SSE / SST)**

## Adjusted R²

> **R²_{\mathrm{adj}}
=
1-
((1-R²)(n-1) / n-1-k)**

## Mathematics habit

Do not memorize formulas without understanding the symbols.

When you see a formula, ask:

```text
What is y?
What is y_hat?
What is the mean?
What is n?
What is k?
What is being squared?
What is being compared?
```

---

# 51. Interview Questions

## Q1. What is Linear Regression?

Linear Regression is a supervised learning algorithm used to model the relationship
between one or more input variables and a continuous numerical target using a linear
equation.

## Q2. What is Simple Linear Regression?

Linear Regression with one input feature.

Example:

```text
CGPA -> Package
```

## Q3. What is the equation of Simple Linear Regression?

> **ŷ=mx+b**

where `m` is slope and `b` is intercept.

## Q4. What is a feature?

A feature is an input variable used by the model to make predictions.

## Q5. What is the target?

The target is the output variable the model is trying to predict.

## Q6. Why do we split data into train and test sets?

To train the model on one portion of the data and evaluate its ability to generalize to
unseen observations.

## Q7. What is `X_train`?

The input features used to train the model.

## Q8. What is `y_train`?

The actual target values corresponding to the training inputs.

## Q9. What is `X_test`?

The input features held out for testing.

## Q10. What is `y_test`?

The actual target values corresponding to the test inputs.

## Q11. What is `y_pred`?

The predictions generated by the trained model for the test inputs.

## Q12. Why use a scatter plot for CGPA vs Package?

Because we want to visualize the relationship between two numerical variables.

## Q13. What is MAE?

Mean Absolute Error. It is the average absolute difference between actual and predicted
values.

## Q14. What is MSE?

Mean Squared Error. It is the average squared difference between actual and predicted
values.

## Q15. Why does MSE penalize large errors more?

Because errors are squared. Larger errors therefore grow faster.

## Q16. Why use RMSE?

RMSE is the square root of MSE, which brings the metric back to the original target
units.

## Q17. Which is more sensitive to large errors, MAE or MSE?

MSE is more sensitive because it squares the errors.

## Q18. What is R²?

R² measures how much variation in the target is explained by the model relative to a
mean baseline.

## Q19. Can R² be negative?

Yes. Negative R² means the model performs worse than the mean baseline under the R²
definition.

## Q20. What does R² = 1 mean?

Perfect predictions for the evaluated observations.

## Q21. What does R² = 0 mean?

The model performs no better than the mean baseline under the R² definition.

## Q22. Is R² accuracy?

No. R² should not simply be called prediction accuracy.

## Q23. What is Adjusted R²?

Adjusted R² modifies R² by accounting for the number of predictors.

## Q24. What are n and k?

```text
n = number of observations
k = number of predictors
```

## Q25. Why is Adjusted R² useful?

It helps compare models while accounting for the number of predictors.

## Q26. Does adding a feature always decrease Adjusted R²?

No. It decreases when the added feature does not improve the model enough to justify the
additional complexity.

## Q27. Why can ordinary R² increase when predictors are added?

Ordinary R² does not decrease simply because additional predictors are included.

## Q28. Why did manual R² match scikit-learn?

Because both calculations use the same underlying mathematical definition.

## Q29. Why can two mathematically equivalent calculations differ by `1e-16`?

Because of floating-point representation and numerical precision.

## Q30. What is the difference between `df['cgpa']` and `df[['cgpa']]`?

The first generally returns a Series. The second returns a one-column DataFrame.

## Q31. Why does scikit-learn expect X to be two-dimensional?

Features are represented as a matrix with the shape:

```text
(number_of_samples, number_of_features)
```

Even one feature therefore has a shape such as:

```text
(200, 1)
```

## Q32. What is the difference between `y_test` and `y_pred`?

`y_test` contains actual target values. `y_pred` contains predictions generated by the
model.

## Q33. What does `model.fit()` do?

It estimates the model parameters from the training data.

## Q34. What does `model.predict()` do?

It uses the learned model parameters to generate predictions for new input observations.

## Q35. Does Linear Regression prove causation?

No. A fitted relationship does not by itself prove that one variable causes another.

---

# 52. Practice Questions

These are the exercises used during the learning process.

## Practice 1: Shape

Suppose a DataFrame has:

```text
5000 rows
20 columns
```

What does this return?

```python
df.shape[1]
```

### Answer

```text
20
```

## Practice 2: Missing Values

Suppose there are:

```text
1000 rows
50 missing values in age
```

How many non-missing age values are there?

### Answer

> **1000-50=950**

## Practice 3: Mean

Given:

```text
5, 5, 5, 5, 5
```

What is the mean?

### Answer

> **5**

## Practice 4: Dispersion

Dataset A:

```text
5, 5, 5, 5, 5
```

Dataset B:

```text
1, 3, 5, 7, 9
```

Which has greater dispersion?

### Answer

Dataset B.

## Practice 5: MAE

Errors:

```text
2, -5, 3, 4
```

Absolute errors:

```text
2, 5, 3, 4
```

Calculate MAE.

### Answer

> **MAE=(2+5+3+4 / 4)=3.5**

## Practice 6: MSE

Squared errors:

```text
4, 25, 9, 16
```

Calculate MSE.

### Answer

> **MSE=(4+25+9+16 / 4)=13.5**

## Practice 7: RMSE

Given:

> **MSE=13.5**

Calculate RMSE.

### Answer

> **RMSE=√{13.5}\approx3.6742**

## Practice 8: R²

Actual values:

```text
10, 20, 30, 40
```

Predicted values:

```text
11, 18, 31, 39
```

Mean:

> **ȳ=25**

SST:

> **SST=500**

SSE:

> **SSE=7**

Calculate R².

### Answer

> **R²=1-(7 / 500)=0.986**

## Practice 9: Perfect R²

Actual:

```text
10, 20, 30, 40
```

Predicted:

```text
10, 20, 30, 40
```

### Answer

> **R²=1**

## Practice 10: Baseline R²

Actual:

```text
10, 20, 30, 40
```

Mean:

```text
25
```

Predicted:

```text
25, 25, 25, 25
```

### Answer

> **R²=0**

## Practice 11: Negative R²

Actual:

```text
10, 20, 30, 40
```

Predicted:

```text
50, 50, 50, 50
```

Given:

> **SSE=3000**

and:

> **SST=500**

Calculate R².

### Answer

> **R²=1-(3000 / 500)=-5**

## Practice 12: Adjusted R²

Given:

> **R²=0.80,\quad n=10,\quad k=2**

### Answer

> **R²_{\mathrm{adj}}\approx0.742857**

## Practice 13: More Predictors

Given:

> **R²=0.80,\quad n=10,\quad k=3**

### Answer

> **R²_{\mathrm{adj}}=0.70**

## Practice 14: Identify n and k

Dataset:

```text
5 rows
CGPA + Age -> Package
```

### Answer

> **n=5,\quad k=2**

---

# 53. Six-Month Revision Checklist

When revisiting this topic after six months, you should be able to answer these without
external help.

## Data

- [ ] What is a DataFrame?
- [ ] What is a Series?
- [ ] What is the difference between `df['cgpa']` and `df[['cgpa']]`?
- [ ] What does `df.shape` return?
- [ ] What does `df.info()` tell us?
- [ ] What does `df.describe()` tell us?
- [ ] What does `.iloc` mean?

## Linear Regression

- [ ] What is supervised learning?
- [ ] What is Linear Regression?
- [ ] What is Simple Linear Regression?
- [ ] What are X and y?
- [ ] What does `ŷ` mean?
- [ ] What is the equation `ŷ = mx + b`?
- [ ] What is slope?
- [ ] What is intercept?
- [ ] What happens during `fit()`?
- [ ] What happens during `predict()`?

## Train and Test

- [ ] Why do we split data?
- [ ] What is training data?
- [ ] What is testing data?
- [ ] What is `X_train`?
- [ ] What is `X_test`?
- [ ] What is `y_train`?
- [ ] What is `y_test`?
- [ ] What is `y_pred`?
- [ ] Why should test data remain unseen during training?

## Metrics

- [ ] MAE formula
- [ ] MSE formula
- [ ] RMSE formula
- [ ] Why use absolute value?
- [ ] Why square errors?
- [ ] Why take the square root for RMSE?
- [ ] Which metric is more sensitive to large errors?
- [ ] What units does each metric use?

## R²

- [ ] What is the mean baseline?
- [ ] What is SST?
- [ ] What is SSE?
- [ ] What is the R² formula?
- [ ] Why can R² be 1?
- [ ] Why can R² be 0?
- [ ] Why can R² be negative?
- [ ] Why is R² not ordinary accuracy?
- [ ] What does R² = 0.773 mean?

## Adjusted R²

- [ ] What is Adjusted R²?
- [ ] What is the formula?
- [ ] What is `n`?
- [ ] What is `k`?
- [ ] Why does Adjusted R² exist?
- [ ] Why can Adjusted R² decrease?
- [ ] When can Adjusted R² increase?
- [ ] When is it useful to compare Adjusted R²?

## Practical workflow

- [ ] Can I load a CSV?
- [ ] Can I inspect a DataFrame?
- [ ] Can I define X and y?
- [ ] Can I split the data?
- [ ] Can I train Linear Regression?
- [ ] Can I make predictions?
- [ ] Can I calculate MAE, MSE and RMSE?
- [ ] Can I calculate R² manually?
- [ ] Can I compare manual results with scikit-learn?
- [ ] Can I explain the model to an interviewer?

---

# 54. Learning Results

The learning block completed the following workflow:

```text
Dataset
   |
   v
Pandas exploration
   |
   v
Statistics
   |
   v
Visualization
   |
   v
Feature and target selection
   |
   v
Train/test split
   |
   v
Model training
   |
   v
Prediction
   |
   v
Actual vs predicted comparison
   |
   v
Residuals
   |
   v
MAE
   |
   v
MSE
   |
   v
RMSE
   |
   v
R²
   |
   v
Adjusted R²
```

The actual CGPA to Package evaluation produced approximately:

```text
R²          = 0.773098
Adjusted R² = 0.767127
```

The R² interpretation is:

> The model explains approximately 77.3% of the variation in package values relative to
> the mean baseline on this evaluation set.

The Adjusted R² is slightly lower because it accounts for the number of predictors.

These results belong to the learning dataset and split used in the notebook.
They should not be treated as a general statement about how well CGPA predicts salary in
the real world.

---

# 55. Next Step

The CGPA to Package example was intentionally simple.

The next stage is to apply Linear Regression to a real-world problem.

The project should demonstrate more than:

```text
Load CSV
   |
   v
fit()
   |
   v
predict()
```

A stronger portfolio project should demonstrate:

```text
Real problem
     |
     v
Business or practical question
     |
     v
Dataset investigation
     |
     v
Data cleaning
     |
     v
Exploratory Data Analysis
     |
     v
Feature engineering
     |
     v
Preprocessing
     |
     v
Baseline
     |
     v
Linear Regression
     |
     v
Evaluation
     |
     v
Error analysis
     |
     v
Interpretation
     |
     v
Limitations
     |
     v
Conclusions
```

The project should demonstrate that we understand why a model is appropriate, not just
how to call `LinearRegression()`.

Future regression topics such as Multiple Linear Regression, assumptions, diagnostics,
regularization, polynomial regression, feature selection, cross-validation and more
advanced regression models belong to later learning blocks.

This README documents the Simple Linear Regression block. It is not the end of
regression or Machine Learning.

---

# 56. Final Mental Model

If you remember one flow, remember this:

```text
                 SIMPLE LINEAR REGRESSION

                       Dataset
                          |
                          v
                  Understand the data
                          |
                          v
                   Visualize X and y
                          |
                          v
                    Choose X and y
                          |
                          v
                   Train/test split
                          |
                          v
                     model.fit()
                          |
                          v
                  Learn model parameters
                          |
                          v
                   model.predict()
                          |
                          v
                Actual vs predicted
                          |
                          v
                  Calculate residuals
                          |
              +-----------+-----------+
              |           |           |
              v           v           v
             MAE         MSE        RMSE
              |           |           |
              +-----------+-----------+
                          |
                          v
                         R²
                          |
                          v
                    Adjusted R²
                          |
                          v
                    Interpretation
```

## The most important distinctions

```text
Notebook
   |
   v
Experiment, run code, inspect outputs and learn

README
   |
   v
Permanent reference and revision document

GitHub
   |
   v
Portfolio and proof of work
```

The notebook records the practical work.

The README records the understanding.

The project demonstrates the ability to apply that understanding to a new problem.

---

## Scope

This README covers the Simple Linear Regression learning block:

- Machine Learning basics;
- Linear Regression;
- Simple Linear Regression;
- Pandas workflow;
- Series and DataFrame;
- visualization;
- feature and target selection;
- train/test split;
- model training;
- prediction;
- residuals;
- MAE;
- MSE;
- RMSE;
- manual metric calculations;
- scikit-learn metrics;
- R²;
- SST;
- SSE;
- mean baseline;
- negative R²;
- Adjusted R²;
- practice problems;
- interview questions;
- common mistakes;
- revision checklist.

Later learning blocks will cover additional regression and Machine Learning topics
separately.