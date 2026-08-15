# Simple Linear Regression — Complete Learning Notes

> **Purpose:** This document is a long-term revision guide for Simple Linear Regression.
>
> It is written so that a beginner with no previous ML exposure can start here, understand the concepts, reproduce the notebook, revise the mathematics, review the important Python/Pandas syntax, and prepare for interviews.
>
> **Learning workflow used in this project:**
>
> **CampusX lecture → handwritten notes → understand the intuition → implement in `.ipynb` → manually verify important mathematics → use `scikit-learn` → evaluate → document in README → build a real-world project.**

---

# Table of Contents

1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [What is Linear Regression?](#2-what-is-linear-regression)
3. [Simple Linear Regression](#3-simple-linear-regression)
4. [The CGPA → Package Problem](#4-the-cgpa--package-problem)
5. [Important Vocabulary](#5-important-vocabulary)
6. [Our Dataset](#6-our-dataset)
7. [Loading the Dataset](#7-loading-the-dataset)
8. [Pandas: Series vs DataFrame](#8-pandas-series-vs-dataframe)
9. [Important Pandas Operations](#9-important-pandas-operations)
10. [Understanding the Data](#10-understanding-the-data)
11. [Descriptive Statistics](#11-descriptive-statistics)
12. [Understanding Mean, Median and Dispersion](#12-understanding-mean-median-and-dispersion)
13. [Standard Deviation — Intuition](#13-standard-deviation--intuition)
14. [Visualization](#14-visualization)
15. [Why a Scatter Plot?](#15-why-a-scatter-plot)
16. [X and y](#16-x-and-y)
17. [Why `df['cgpa']` and `df[['cgpa']]` Are Different](#17-why-dfcgpa-and-dfcgpa-are-different)
18. [Train/Test Split](#18-traintest-split)
19. [Why We Split the Data](#19-why-we-split-the-data)
20. [X_train, X_test, y_train, y_test](#20-x_train-x_test-y_train-y_test)
21. [Training a Linear Regression Model](#21-training-a-linear-regression-model)
22. [What Training Actually Means](#22-what-training-actually-means)
23. [The Linear Regression Equation](#23-the-linear-regression-equation)
24. [Slope and Intercept](#24-slope-and-intercept)
25. [Prediction](#25-prediction)
26. [Actual vs Predicted Values](#26-actual-vs-predicted-values)
27. [Comparison Table](#27-comparison-table)
28. [Residual / Error](#28-residual--error)
29. [MAE](#29-mae)
30. [MSE](#30-mse)
31. [RMSE](#31-rmse)
32. [MAE vs MSE vs RMSE](#32-mae-vs-mse-vs-rmse)
33. [Manual Metrics vs scikit-learn](#33-manual-metrics-vs-scikit-learn)
34. [Floating-Point Precision](#34-floating-point-precision)
35. [R² Score](#35-r2-score)
36. [The Mean Baseline](#36-the-mean-baseline)
37. [SST](#37-sst)
38. [SSE](#38-sse)
39. [Understanding the R² Formula](#39-understanding-the-r2-formula)
40. [R² = 1, 0 and Negative R²](#40-r2--1-0-and-negative-r2)
41. [R² on the Real Dataset](#41-r2-on-the-real-dataset)
42. [Adjusted R²](#42-adjusted-r2)
43. [n and k](#43-n-and-k)
44. [Adjusted R² Mathematics](#44-adjusted-r2-mathematics)
45. [R² vs Adjusted R²](#45-r2-vs-adjusted-r2)
46. [Complete Notebook Workflow](#46-complete-notebook-workflow)
47. [Important Code Snippets to Memorize](#47-important-code-snippets-to-memorize)
48. [Common Mistakes](#48-common-mistakes)
49. [Maths Revision Sheet](#49-maths-revision-sheet)
50. [Interview Questions and Answers](#50-interview-questions-and-answers)
51. [Practice Questions](#51-practice-questions)
52. [Six-Month Revision Checklist](#52-six-month-revision-checklist)
53. [What We Have Completed](#53-what-we-have-completed)
54. [Next Step: Real-World Project](#54-next-step-real-world-project)

---

# 1. What is Machine Learning?

Machine Learning is a way of building systems that learn patterns from data and use those patterns to make predictions or decisions.

A traditional program often looks like:

```text
Rules + Data → Output
```

A machine learning workflow can be thought of as:

```text
Data + Correct Outputs
        ↓
     Learning
        ↓
      Model
        ↓
New Data → Prediction
```

For our problem:

```text
Student data
    ↓
CGPA + Package examples
    ↓
Linear Regression learns relationship
    ↓
New student's CGPA
    ↓
Predicted package
```

---

# 2. What is Linear Regression?

Linear Regression is a supervised machine learning algorithm used to model a relationship between input variable(s) and a numerical target.

The basic idea is:

> Find a line that represents the relationship between the input and output as well as possible.

For Simple Linear Regression:

```text
One input feature → One numerical target
```

Example:

```text
CGPA → Package
```

The model tries to find a straight line through the data.

---

# 3. Simple Linear Regression

"Simple" means that there is **one independent/input variable**.

Example:

```text
X = CGPA
y = Package
```

The model is:

\[
\hat y = mx + b
\]

Where:

- \(x\) = input feature
- \(\hat y\) = predicted output
- \(m\) = slope
- \(b\) = intercept

For our problem:

\[
\widehat{Package}=m(CGPA)+b
\]

The model learns suitable values of \(m\) and \(b\).

---

# 4. The CGPA → Package Problem

Our dataset contains:

```text
cgpa
package
```

We want to learn:

```text
CGPA → Package
```

The intuition is:

> Students with higher CGPA may generally receive higher packages, although the relationship will not be perfect.

This makes Simple Linear Regression a good learning example because:

- there is one input feature;
- the target is numerical;
- the relationship appears approximately linear;
- we can visualize it easily;
- we can understand the mathematics manually.

---

# 5. Important Vocabulary

## Feature / Input / Independent Variable

The information given to the model to make a prediction.

Here:

```text
CGPA
```

## Target / Output / Dependent Variable

The value we want to predict.

Here:

```text
Package
```

## Observation / Instance / Sample

One row of data.

Example:

```text
CGPA = 7.82
Package = 3.7
```

That is one observation.

## Model

The mathematical relationship learned from training data.

## Training

Giving training data to the algorithm so it can learn parameters/patterns.

## Testing

Evaluating the trained model on data it did not use during training.

## Prediction

The output generated by the trained model for an input.

---

# 6. Our Dataset

The dataset has:

```text
200 rows
2 columns
```

Columns:

```text
cgpa
package
```

We observed from `df.info()` that both columns are:

```text
float64
```

and both contained:

```text
200 non-null
```

values.

Therefore, for the dataset we inspected:

- 200 observations
- no missing values in `cgpa`
- no missing values in `package`

---

# 7. Loading the Dataset

```python
import pandas as pd

df = pd.read_csv("../data/placement.csv")
```

## What does this mean?

```python
pd.read_csv(...)
```

means:

> Use Pandas to read a CSV file.

```text
../
```

means:

> Go one directory up from the current location.

Then:

```text
data/
    placement.csv
```

is inside that parent directory.

So:

```python
df = pd.read_csv("../data/placement.csv")
```

means:

> Read `placement.csv` and store the resulting DataFrame in the variable `df`.

### Important

`df` is simply a variable name. It commonly means "DataFrame", but Python does not require that name.

You could technically write:

```python
data = pd.read_csv("../data/placement.csv")
```

---

# 8. Pandas: Series vs DataFrame

This caused important confusion during learning, so remember it carefully.

## Series

A single column is usually a Pandas `Series`.

```python
df['cgpa']
```

returns one-dimensional data:

```text
0      6.89
1      5.12
2      7.82
...
199    6.22
Name: cgpa
Length: 200
dtype: float64
```

Think:

```text
Series
↓
ONE column
↓
1-dimensional
```

## DataFrame

A DataFrame is a table.

```python
df[['cgpa']]
```

returns:

```text
     cgpa
0    6.89
1    5.12
2    7.82
...
```

Think:

```text
DataFrame
↓
TABLE
↓
2-dimensional
```

Even if the table has only one column, it is still a DataFrame.

---

# 9. Important Pandas Operations

## Shape

```python
df.shape
```

returns:

```text
(rows, columns)
```

Our dataset:

```text
(200, 2)
```

Therefore:

```python
df.shape[0]
```

→ number of rows:

```text
200
```

and:

```python
df.shape[1]
```

→ number of columns:

```text
2
```

### Memory trick

```text
shape[0] → rows
shape[1] → columns
```

---

## Select one column

```python
df['cgpa']
```

Returns a Series.

---

## Select one column as a DataFrame

```python
df[['cgpa']]
```

Returns a DataFrame.

---

## Select multiple columns

```python
df[['cgpa', 'package']]
```

Returns a DataFrame.

---

## Select rows/columns using `.iloc`

`.iloc` means integer-location based indexing.

Example:

```python
df.iloc[0]
```

means:

> Give me row at integer position 0.

```python
df.iloc[0:5]
```

means:

> Give me rows from position 0 up to, but not including, 5.

Example:

```python
df.iloc[:, 0]
```

means:

```text
:
↓
all rows

0
↓
first column
```

So:

```python
df.iloc[:, 0]
```

means:

> All rows from the first column.

### Remember

```text
.iloc[row_position, column_position]
```

---

# 10. Understanding the Data

## `shape`

```python
df.shape
```

Answers:

> How many rows and columns do I have?

---

## `info()`

```python
df.info()
```

Tells us useful structural information:

- DataFrame type
- number of rows
- column names
- non-null counts
- data types
- memory usage

Example information we observed:

```text
RangeIndex: 200 entries
Data columns: 2
cgpa: 200 non-null, float64
package: 200 non-null, float64
```

### Important interview question

**What is the difference between `shape` and `info()`?**

`shape` gives the dimensions.

`info()` gives a broader structural summary including data types and non-null counts.

---

## `describe()`

```python
df.describe()
```

Gives descriptive statistics for numerical columns.

We observed:

```text
cgpa:
mean ≈ 6.99
std  ≈ 1.07
min  = 4.26
max  = 9.58

package:
mean ≈ 3.00
std  ≈ 0.69
min  = 1.37
max  = 4.62
```

---

# 11. Descriptive Statistics

Important values from `describe()`:

## Count

Number of non-missing observations.

## Mean

Arithmetic average:

\[
Mean=\frac{\text{sum of values}}{\text{number of values}}
\]

Example:

```text
5, 5, 5, 5, 5
```

\[
Mean=\frac{25}{5}=5
\]

## Median

Middle value after sorting the data.

If:

```text
1, 2, 3, 4, 5
```

median = 3.

If:

```text
1, 2, 3, 4
```

median:

\[
\frac{2+3}{2}=2.5
\]

## Minimum

Smallest value.

## Maximum

Largest value.

## 25%

First quartile, also called Q1.

## 50%

Median, also called Q2.

## 75%

Third quartile, also called Q3.

---

# 12. Understanding Mean, Median and Dispersion

A useful intuition:

```text
5, 5, 5, 5, 5
```

has:

```text
Mean = 5
Median = 5
```

and the values have essentially no spread around the mean.

Compare with:

```text
1, 3, 5, 7, 9
```

Mean:

\[
5
\]

Median:

\[
5
\]

but the values are much more spread out.

This teaches an important point:

> Mean and median describe the center, but they do not by themselves tell us how spread out the data is.

That is where measures such as standard deviation become useful.

---

# 13. Standard Deviation — Intuition

Our `describe()` output showed approximately:

```text
CGPA std     ≈ 1.069
Package std  ≈ 0.692
```

Standard deviation measures the typical amount of spread around the mean.

A beginner-friendly intuition:

> A larger standard deviation means the observations tend to be more spread out around their mean.

For our data, CGPA has greater numerical dispersion than package according to their standard deviations.

### Important correction

Do not describe standard deviation as:

> "Every point is exactly 1.07 away from the mean."

That is not what standard deviation means.

It is a measure of overall spread, not a guarantee that every observation is that distance from the mean.

---

# 14. Visualization

We used Matplotlib:

```python
import matplotlib.pyplot as plt
```

We then created a scatter plot:

```python
plt.scatter(df['cgpa'], df['package'])

plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('CGPA vs Package')

plt.show()
```

---

# 15. Why a Scatter Plot?

We are comparing:

```text
one numerical feature
        vs
one numerical target
```

Specifically:

```text
CGPA vs Package
```

A scatter plot places each observation as a point:

```text
x-axis → CGPA
y-axis → Package
```

This lets us visually inspect whether there is a relationship.

We observed:

1. Package generally increased from left to right.
2. The points could roughly be represented by a straight line.
3. The points were not perfectly on one line.
4. There was visible noise.

This made Simple Linear Regression a reasonable model to explore.

### Why not a bar chart?

A bar chart is generally better for comparing discrete categories.

Our CGPA/package observations are paired numerical observations, so a scatter plot is more informative.

### Why not a histogram?

A histogram shows the distribution of one numerical variable. It does not directly show the relationship between CGPA and package.

### Why not a line chart?

A line chart implies an ordered sequence where connecting adjacent points has meaning. Our student observations are not a time series.

---

# 16. X and y

For our model:

```python
X = df[['cgpa']]
y = df['package']
```

## X

`X` contains the input/features.

```text
CGPA
```

## y

`y` contains the target/output.

```text
Package
```

Think:

```text
X
↓
Input

Model
↓

y
↓
Output
```

---

# 17. Why `df['cgpa']` and `df[['cgpa']]` Are Different

This is extremely important in Pandas and scikit-learn.

```python
df['cgpa']
```

returns a Series:

```text
1D
```

Shape:

```python
df['cgpa'].shape
```

would be:

```text
(200,)
```

But:

```python
df[['cgpa']]
```

returns a DataFrame:

```text
2D
```

Shape:

```python
df[['cgpa']].shape
```

is:

```text
(200, 1)
```

### Why does this matter?

Machine learning libraries often expect features as a **2-dimensional matrix**:

```text
(number_of_samples, number_of_features)
```

So for one feature and 200 observations:

```text
(200, 1)
```

is the natural feature-matrix shape.

For the target, a 1D structure is commonly acceptable:

```text
(200,)
```

Therefore:

```python
X = df[['cgpa']]
y = df['package']
```

is a very common pattern.

---

# 18. Train/Test Split

We used:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

With 200 observations and `test_size=0.2`:

```text
80% → training
20% → testing
```

So approximately:

```text
Training = 160
Testing  = 40
```

---

# 19. Why We Split the Data

The model needs to be evaluated on data it did not use for training.

If we trained and evaluated on exactly the same examples, we would not get a useful estimate of performance on unseen data.

The idea is:

```text
Training data
     ↓
Model learns
     ↓
Testing data
     ↓
Model predicts
     ↓
Evaluate predictions
```

### Important correction

Do not say:

> "We split the data because otherwise the model will memorize everything and give 100% accuracy."

That is too simplistic.

The more accurate idea is:

> We separate training and testing data so that we can evaluate how well the trained model generalizes to unseen observations.

---

# 20. X_train, X_test, y_train, y_test

```text
X_train
```

Input features used for training.

```text
X_test
```

Input features held back for testing.

```text
y_train
```

Actual target values corresponding to training inputs.

```text
y_test
```

Actual target values corresponding to test inputs.

A useful table:

| Variable | Meaning |
|---|---|
| X_train | Training inputs |
| y_train | Training targets |
| X_test | Unseen test inputs |
| y_test | Actual test targets |

Important:

> `y_test` is **not** the model's prediction.

It contains the actual values.

The model's predictions are stored in:

```python
y_pred
```

---

# 21. Training a Linear Regression Model

Import:

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

---

# 22. What Training Actually Means

When we write:

```python
model.fit(X_train, y_train)
```

we are saying:

> Learn the relationship between the training CGPA values and the corresponding package values.

The model learns parameters of the line.

For Simple Linear Regression:

\[
\hat y=mx+b
\]

The learned values are:

```text
m → slope
b → intercept
```

Training is **not** the same as human-style memorization.

The important idea is that the algorithm finds parameters that minimize an objective/loss according to the training procedure.

---

# 23. The Linear Regression Equation

The basic equation is:

\[
\boxed{\hat y=mx+b}
\]

For our problem:

\[
\boxed{\widehat{Package}=m(CGPA)+b}
\]

Where:

- \(m\) = slope
- \(b\) = intercept
- \(x\) = CGPA
- \(\hat y\) = predicted package

---

# 24. Slope and Intercept

## Slope

The slope tells us how much the predicted output changes when the input increases by one unit.

If:

\[
m=0.5
\]

then an increase of 1 CGPA point corresponds to a predicted package increase of 0.5 package units, according to the fitted line.

## Intercept

The intercept is the predicted value when:

\[
x=0
\]

For a mathematical line:

\[
\hat y=m(0)+b=b
\]

### Important caution

The intercept does not always have a useful real-world interpretation.

If CGPA = 0 is outside the meaningful range of the data, interpreting the intercept literally may not make sense.

---

# 25. Prediction

After training:

```python
y_pred = model.predict(X_test)
```

This means:

> Give the trained model the unseen test CGPA values and ask it to predict their package values.

For example, `y_pred` might contain:

```text
[2.7803, 3.1363, 3.1995, ...]
```

Each value corresponds to one row of `X_test`.

---

# 26. Actual vs Predicted Values

We have:

```text
y_test
↓
Actual package

y_pred
↓
Predicted package
```

These can be compared.

Example:

```text
Actual       Predicted
3.00         2.78
3.20         3.14
3.50         3.20
...
```

The difference between them is the prediction error/residual.

---

# 27. Comparison Table

We created:

```python
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print(comparison)
```

## Line by line

```python
pd.DataFrame(...)
```

creates a DataFrame/table.

```python
'Actual': y_test.values
```

creates a column called `Actual`.

```python
'Predicted': y_pred
```

creates a column called `Predicted`.

## Why `y_test.values`?

`y_test` is a Pandas Series.

`.values` extracts its underlying values in an array-like form.

We did not need:

```python
y_pred.values
```

because `y_pred` returned by `model.predict()` is already a NumPy array.

So:

```python
y_test.values
```

is reasonable here, while:

```python
y_pred.values
```

would generally fail because `y_pred` is not a Pandas Series/DataFrame.

---

# 28. Residual / Error

We created:

```python
comparison['Error'] = (
    comparison['Actual'] - comparison['Predicted']
)
```

This creates a new column called `Error`.

For one row:

```text
Actual = 3.5
Predicted = 3.2

Error = 3.5 - 3.2
      = 0.3
```

If:

```text
Actual = 3.2
Predicted = 3.5
```

then:

```text
Error = -0.3
```

The sign tells us the direction of the error.

- Positive error → actual > predicted
- Negative error → actual < predicted
- Zero → exact prediction

For many evaluation metrics, we transform these errors so positive and negative errors do not cancel.

---

# 29. MAE

MAE = Mean Absolute Error.

Formula:

\[
\boxed{
MAE=\frac{1}{n}\sum |y_i-\hat y_i|
}
\]

In simple words:

> Find every prediction error, ignore its sign using absolute value, then take the average.

Example:

```text
Errors:
2, -5, 3, 4
```

Absolute errors:

```text
2, 5, 3, 4
```

Sum:

\[
14
\]

MAE:

\[
\frac{14}{4}=3.5
\]

## Python

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(mae)
```

### Intuition

If:

```text
MAE = 0.23
```

the model's average absolute prediction error is about 0.23 target units.

MAE is easy to interpret because it stays in the original target units.

---

# 30. MSE

MSE = Mean Squared Error.

Formula:

\[
\boxed{
MSE=\frac{1}{n}\sum(y_i-\hat y_i)^2
}
\]

Steps:

1. Calculate errors.
2. Square them.
3. Average them.

Example errors:

```text
2, -5, 3, 4
```

Squared errors:

```text
4, 25, 9, 16
```

Sum:

\[
54
\]

MSE:

\[
\frac{54}{4}=13.5
\]

## Why square?

Squaring:

1. removes the negative sign;
2. gives larger errors much greater influence.

Therefore, MSE is more sensitive to large errors/outliers than MAE.

---

# 31. RMSE

RMSE = Root Mean Squared Error.

Formula:

\[
\boxed{
RMSE=\sqrt{MSE}
}
\]

For the previous example:

\[
MSE=13.5
\]

Therefore:

\[
RMSE=\sqrt{13.5}
\]

\[
RMSE\approx3.6742
\]

## Why take the square root?

MSE is in squared units.

If package is measured in package units, MSE is in package-squared units.

Taking the square root brings the metric back to the original unit.

This makes RMSE easier to interpret.

---

# 32. MAE vs MSE vs RMSE

| Metric | Main idea | Penalizes large errors? | Original units? |
|---|---|---|---|
| MAE | Average absolute error | Less strongly | Yes |
| MSE | Average squared error | Strongly | No |
| RMSE | Square root of MSE | Strongly | Yes |

### Memory trick

```text
MAE
↓
Absolute

MSE
↓
Square

RMSE
↓
Square + Root
```

---

# 33. Manual Metrics vs scikit-learn

We manually calculated metrics and compared them with scikit-learn.

For MAE, for example:

```python
mae = comparison['Absolute Error'].mean()
```

and:

```python
from sklearn.metrics import mean_absolute_error

mae_sklearn = mean_absolute_error(y_test, y_pred)
```

We found:

```text
Difference = 0.0
```

This is useful because it verifies that our understanding and implementation match the library's calculation.

### Why use sklearn in real projects?

Because:

- it is standard;
- it is tested;
- it is concise;
- it reduces implementation mistakes.

### Why learn the manual calculation?

Because a good ML practitioner should understand what the metric actually measures.

---

# 34. Floating-Point Precision

When comparing manual and sklearn R², we got:

```text
Difference:
1.1102230246251565e-16
```

This looks strange, but it is effectively zero.

It is a tiny floating-point representation/calculation difference.

You can check:

```python
np.isclose(r2_manual, r2_sklearn)
```

which should return:

```text
True
```

Do not interpret a value around \(10^{-16}\) as meaningful model error.

---

# 35. R² Score

R² = coefficient of determination.

Formula:

\[
\boxed{
R^2=1-\frac{SSE}{SST}
}
\]

R² compares our model against a simple baseline that predicts the mean.

The key intuition:

> **How much better is our model than simply predicting the mean?**

---

# 36. The Mean Baseline

Suppose actual values are:

```text
10, 20, 30, 40
```

Mean:

\[
\frac{10+20+30+40}{4}=25
\]

A very simple baseline model could predict:

```text
25
25
25
25
```

for every observation.

This baseline is important because R² asks how our model compares with this simple strategy.

---

# 37. SST

SST = Total Sum of Squares.

Formula:

\[
\boxed{
SST=\sum(y_i-\bar y)^2
}
\]

Intuition:

> How much total variation exists in the actual target values around their mean?

Our tiny example:

```text
Actual = [10, 20, 30, 40]
Mean = 25
```

Then:

\[
(10-25)^2+(20-25)^2+(30-25)^2+(40-25)^2
\]

\[
=225+25+25+225
\]

\[
\boxed{SST=500}
\]

Python:

```python
sst = sum((y - mean) ** 2 for y in actual)
```

---

# 38. SSE

SSE = Sum of Squared Errors.

Formula:

\[
\boxed{
SSE=\sum(y_i-\hat y_i)^2
}
\]

Intuition:

> How much squared prediction error does our actual model make?

Python:

```python
sse = sum(
    (y - y_pred_value) ** 2
    for y, y_pred_value in zip(actual, predicted)
)
```

---

# 39. Understanding the R² Formula

\[
R^2=1-\frac{SSE}{SST}
\]

Remember:

```text
SST → Error of the mean baseline

SSE → Error of our model
```

So R² asks:

> How much of the baseline's error did our model eliminate?

### Why `1 -`?

If model error is zero:

\[
SSE=0
\]

then:

\[
R^2=1
\]

If model error equals baseline error:

\[
SSE=SST
\]

then:

\[
R^2=0
\]

If model error is larger than baseline error:

\[
SSE>SST
\]

then:

\[
R^2<0
\]

---

# 40. R² = 1, 0 and Negative R²

We deliberately tested all three cases.

## Perfect prediction

```text
Actual:
10, 20, 30, 40

Predicted:
10, 20, 30, 40
```

All errors are zero.

\[
SSE=0
\]

Therefore:

\[
\boxed{R^2=1}
\]

Meaning:

> Perfect predictions for this evaluation.

---

## Same as mean baseline

```text
Predicted:
25, 25, 25, 25
```

Since 25 is the mean:

\[
SSE=SST
\]

Therefore:

\[
\boxed{R^2=0}
\]

Meaning:

> The model performs no better than the mean baseline under this metric.

---

## Worse than the baseline

```text
Predicted:
50, 50, 50, 50
```

We calculated:

\[
SSE=3000
\]

and:

\[
SST=500
\]

Therefore:

\[
R^2=1-\frac{3000}{500}
\]

\[
\boxed{R^2=-5}
\]

Meaning:

> The model performs worse than the mean baseline.

### Important

Never describe R² as ordinary "accuracy."

For example:

```text
R² = 0.773
```

does NOT mean:

```text
77.3% accuracy
```

---

# 41. R² on the Real Dataset

Our actual CGPA → Package model produced:

```text
Manual R²:
0.7730984312051674

Sklearn R²:
0.7730984312051673
```

Difference:

```text
1.1102230246251565e-16
```

These are effectively the same.

Therefore:

\[
\boxed{R^2\approx0.7731}
\]

Interpretation:

> Our Linear Regression model explains approximately 77.3% of the variation in package values relative to the mean baseline.

Do not say:

> "The model is 77.3% accurate."

---

# 42. Adjusted R²

Adjusted R² modifies R² by accounting for the number of predictors/features used by the model.

Formula:

\[
\boxed{
R^2_{adj}
=
1-
\frac{(1-R^2)(n-1)}
{n-1-k}
}
\]

The intuition:

> R² rewards explanatory power. Adjusted R² also considers model complexity.

---

# 43. n and k

This distinction is extremely important.

## n

Number of observations/rows used in the evaluation.

## k

Number of independent/input variables/features.

Example:

```text
5 rows
CGPA + Age → Package
```

Then:

\[
n=5
\]

and:

\[
k=2
\]

### Our current evaluation

Our test set contains:

```text
40 observations
```

Therefore:

\[
n=40
\]

Our model:

```text
CGPA → Package
```

has one predictor:

\[
k=1
\]

---

# 44. Adjusted R² Mathematics

Example:

\[
R^2=0.80,\quad n=10,\quad k=2
\]

Formula:

\[
R^2_{adj}
=
1-
\frac{(1-0.80)(10-1)}
{10-1-2}
\]

\[
=1-\frac{0.20\times9}{7}
\]

\[
=1-\frac{1.8}{7}
\]

\[
\boxed{R^2_{adj}\approx0.742857}
\]

When we changed:

\[
k=3
\]

while keeping:

\[
R^2=0.80,\quad n=10
\]

we got:

\[
\boxed{R^2_{adj}=0.70}
\]

This demonstrates the complexity penalty.

### Important nuance

Do not say:

> "Adjusted R² always decreases when a feature is added."

The correct statement is:

> Adjusted R² can decrease if the new feature does not improve the model enough to justify the added complexity.

---

# 45. R² vs Adjusted R²

| R² | Adjusted R² |
|---|---|
| Measures explained variation relative to mean baseline | Similar, but accounts for number of predictors |
| Does not decrease simply because predictors are added | Can decrease when unnecessary predictors are added |
| Useful for measuring explanatory performance | Useful when comparing models with different numbers of predictors |
| Simpler | Includes a complexity adjustment |

Memory:

```text
R²
↓
How much variation is explained?

Adjusted R²
↓
How much variation is explained
after accounting for predictors?
```

---

# 46. Complete Notebook Workflow

Our `.ipynb` follows this general structure:

```text
Simple Linear Regression
│
├── 1. Import libraries
│
├── 2. Load dataset
│
├── 3. Explore dataset
│     ├── shape
│     ├── info()
│     └── describe()
│
├── 4. Understand statistics
│
├── 5. Visualize
│     └── scatter plot
│
├── 6. Define X and y
│
├── 7. Train/test split
│
├── 8. Train Linear Regression
│
├── 9. Predict
│
├── 10. Comparison table
│
├── 11. Error
│
├── 12. MAE
│
├── 13. MSE
│
├── 14. RMSE
│
├── 15. R²
│     ├── tiny dataset
│     └── real dataset
│
└── 16. Adjusted R²
      ├── tiny mathematical examples
      └── real dataset
```

This separation is deliberate:

```text
Tiny dataset
↓
Understand mathematics

Real dataset
↓
Apply ML workflow
```

---

# 47. Important Code Snippets to Memorize

## Load CSV

```python
df = pd.read_csv("../data/placement.csv")
```

## Dimensions

```python
df.shape
```

```python
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
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

## Create model

```python
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

## MAE

```python
mae = mean_absolute_error(y_test, y_pred)
```

## MSE

```python
mse = mean_squared_error(y_test, y_pred)
```

## RMSE

```python
rmse = np.sqrt(mse)
```

## R²

```python
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

# 48. Common Mistakes

## Mistake 1: Confusing Series and DataFrame

```python
df['cgpa']
```

is usually a Series.

```python
df[['cgpa']]
```

is a DataFrame.

---

## Mistake 2: Thinking `y_test` is prediction

Wrong:

```text
y_test = prediction
```

Correct:

```text
y_test = actual unseen target values
y_pred = model predictions
```

---

## Mistake 3: Calling R² accuracy

Wrong:

> R² = 77% accuracy.

Better:

> R² = 0.77 means the model explains approximately 77% of the variation relative to the mean baseline.

---

## Mistake 4: Thinking MAE removes all error when one absolute error is zero

If one error is zero, that particular prediction is perfect.

It does not mean the entire model has zero error.

---

## Mistake 5: Forgetting why MSE squares

Squaring prevents positive and negative errors from cancelling and gives large errors more influence.

---

## Mistake 6: Thinking RMSE is unrelated to MSE

RMSE is simply:

\[
RMSE=\sqrt{MSE}
\]

---

## Mistake 7: Thinking negative R² is impossible

It is possible.

It means the model performs worse than the mean baseline under the R² definition.

---

## Mistake 8: Saying standard deviation means every point is that far from the mean

Standard deviation measures overall spread; it does not mean every observation is exactly that distance away.

---

## Mistake 9: Confusing n and k

```text
n → number of observations
k → number of predictors/features
```

---

## Mistake 10: Evaluating only on training data

Training performance alone does not tell us how well the model generalizes to unseen observations.

---

# 49. Maths Revision Sheet

These are the formulas worth keeping comfortable with.

## Mean

\[
\boxed{
\bar{x}=\frac{\sum x_i}{n}
}
\]

## Linear equation

\[
\boxed{
y=mx+b
}
\]

## Prediction

\[
\boxed{
\hat y=mx+b
}
\]

## Error / residual

\[
\boxed{
e_i=y_i-\hat y_i
}
\]

## MAE

\[
\boxed{
MAE=\frac{1}{n}\sum|y_i-\hat y_i|
}
\]

## MSE

\[
\boxed{
MSE=\frac{1}{n}\sum(y_i-\hat y_i)^2
}
\]

## RMSE

\[
\boxed{
RMSE=\sqrt{MSE}
}
\]

## SST

\[
\boxed{
SST=\sum(y_i-\bar y)^2
}
\]

## SSE

\[
\boxed{
SSE=\sum(y_i-\hat y_i)^2
}
\]

## R²

\[
\boxed{
R^2=1-\frac{SSE}{SST}
}
\]

## Adjusted R²

\[
\boxed{
R^2_{adj}
=
1-
\frac{(1-R^2)(n-1)}
{n-1-k}
}
\]

### Maths habit

Do not try to memorize formulas without understanding the symbols.

Always ask:

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

# 50. Interview Questions and Answers

## Q1. What is Linear Regression?

Linear Regression is a supervised learning algorithm used to model the relationship between input variable(s) and a continuous numerical target using a linear equation.

---

## Q2. What is Simple Linear Regression?

Linear Regression with one independent/input variable.

Example:

```text
CGPA → Package
```

---

## Q3. What is the equation of Simple Linear Regression?

\[
\hat y=mx+b
\]

where \(m\) is slope and \(b\) is intercept.

---

## Q4. What is a feature?

An input variable used by the model to make predictions.

---

## Q5. What is the target?

The output variable the model is trying to predict.

---

## Q6. Why do we split data into train and test sets?

To train the model on one portion and evaluate its ability to generalize to unseen observations using another portion.

---

## Q7. What is `X_train`?

Input features used for training.

---

## Q8. What is `y_train`?

Actual target values corresponding to the training inputs.

---

## Q9. What is `X_test`?

Input features held out for evaluation.

---

## Q10. What is `y_test`?

Actual target values corresponding to the test inputs.

---

## Q11. What is `y_pred`?

The predictions generated by the trained model for the test inputs.

---

## Q12. Why do we use a scatter plot for CGPA vs Package?

Because we want to visualize the relationship between two numerical variables.

---

## Q13. What is MAE?

Mean Absolute Error. It is the average absolute difference between actual and predicted values.

---

## Q14. What is MSE?

Mean Squared Error. It is the average squared difference between actual and predicted values.

---

## Q15. Why does MSE penalize large errors more?

Because errors are squared. Larger errors grow faster when squared.

---

## Q16. Why use RMSE?

RMSE is the square root of MSE, bringing the metric back to the original target units.

---

## Q17. Which is more sensitive to outliers: MAE or MSE?

MSE is more sensitive because it squares the errors.

---

## Q18. What is R²?

R² measures how much variation in the target is explained by the model relative to a mean baseline.

---

## Q19. Can R² be negative?

Yes. A negative R² means the model performs worse than the mean baseline under the R² criterion.

---

## Q20. What does R² = 1 mean?

Perfect predictions for the evaluated observations.

---

## Q21. What does R² = 0 mean?

The model performs no better than the mean baseline under the R² definition.

---

## Q22. Is R² accuracy?

No. R² should not simply be called prediction accuracy.

---

## Q23. What is Adjusted R²?

Adjusted R² modifies R² to account for the number of predictors used by the model.

---

## Q24. What are n and k in Adjusted R²?

```text
n = number of observations
k = number of predictors/features
```

---

## Q25. Why is Adjusted R² useful?

It helps compare models while accounting for the number of predictors and can penalize unnecessary features.

---

## Q26. Does adding a feature always decrease Adjusted R²?

No. It decreases when the added feature does not improve the model enough to justify the added complexity.

---

## Q27. Why can R² increase when adding predictors?

Ordinary R² does not decrease simply because additional predictors are included.

---

## Q28. Why did our manual R² match sklearn?

Because we implemented the same underlying mathematical definition.

---

## Q29. Why might two mathematically equivalent calculations differ by `1e-16`?

Floating-point representation and numerical precision.

---

## Q30. What is the difference between `df['cgpa']` and `df[['cgpa']]`?

The first generally returns a Series; the second returns a one-column DataFrame.

---

# 51. Practice Questions

These are the questions we worked through during learning.

---

## Practice 1 — Shape

Suppose:

```text
5000 rows
20 columns
```

What is:

```python
df.shape[1]
```

### Answer

```text
20
```

---

## Practice 2 — Missing values

Suppose:

```text
1000 rows
50 missing values in age
```

How many non-missing age values?

### Answer

\[
1000-50=950
\]

---

## Practice 3 — Mean

Given:

```text
5, 5, 5, 5, 5
```

What is the mean?

### Answer

\[
5
\]

---

## Practice 4 — Dispersion

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

B.

---

## Practice 5 — MAE

Errors:

```text
2, -5, 3, 4
```

Absolute errors:

```text
2, 5, 3, 4
```

MAE:

\[
\boxed{3.5}
\]

---

## Practice 6 — MSE

Squared errors:

```text
4, 25, 9, 16
```

MSE:

\[
\frac{54}{4}=13.5
\]

### Answer

\[
\boxed{13.5}
\]

---

## Practice 7 — RMSE

\[
RMSE=\sqrt{13.5}
\]

### Answer

\[
\boxed{3.6742\text{ approximately}}
\]

---

## Practice 8 — R²

Actual:

```text
10, 20, 30, 40
```

Predicted:

```text
11, 18, 31, 39
```

Mean:

\[
25
\]

SST:

\[
500
\]

SSE:

\[
7
\]

R²:

\[
1-\frac{7}{500}
\]

### Answer

\[
\boxed{R^2=0.986}
\]

---

## Practice 9 — Perfect R²

Actual:

```text
10, 20, 30, 40
```

Predicted:

```text
10, 20, 30, 40
```

### Answer

\[
\boxed{R^2=1}
\]

---

## Practice 10 — Baseline R²

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

\[
\boxed{R^2=0}
\]

---

## Practice 11 — Negative R²

Actual:

```text
10, 20, 30, 40
```

Predicted:

```text
50, 50, 50, 50
```

We obtained:

\[
SSE=3000
\]

\[
SST=500
\]

Therefore:

\[
R^2=1-\frac{3000}{500}
\]

### Answer

\[
\boxed{-5}
\]

---

## Practice 12 — Adjusted R²

Given:

\[
R^2=0.80,\quad n=10,\quad k=2
\]

### Answer

\[
R^2_{adj}\approx0.742857
\]

---

## Practice 13 — More predictors

Given:

\[
R^2=0.80,\quad n=10,\quad k=3
\]

### Answer

\[
R^2_{adj}=0.70
\]

---

## Practice 14 — Identify n and k

Dataset:

```text
5 rows
CGPA + Age → Package
```

### Answer

\[
n=5,\quad k=2
\]

---

# 52. Six-Month Revision Checklist

When revisiting this topic after six months, you should be able to answer these without external help:

### Data

- [ ] What is a DataFrame?
- [ ] What is a Series?
- [ ] Difference between `df['cgpa']` and `df[['cgpa']]`
- [ ] What does `df.shape` return?
- [ ] What does `df.info()` tell us?
- [ ] What does `df.describe()` tell us?
- [ ] What does `.iloc` mean?

### Linear Regression

- [ ] What is supervised learning?
- [ ] What is Simple Linear Regression?
- [ ] What are X and y?
- [ ] What is the equation \(y=mx+b\)?
- [ ] What is slope?
- [ ] What is intercept?
- [ ] What happens during `.fit()`?
- [ ] What happens during `.predict()`?

### Train/Test

- [ ] Why split data?
- [ ] Difference between training and testing?
- [ ] Difference between X_train and X_test?
- [ ] Difference between y_test and y_pred?

### Metrics

- [ ] MAE formula
- [ ] MSE formula
- [ ] RMSE formula
- [ ] Why square errors?
- [ ] Why take the square root for RMSE?
- [ ] Which metric is more sensitive to large errors?

### R²

- [ ] What is the mean baseline?
- [ ] What is SST?
- [ ] What is SSE?
- [ ] R² formula
- [ ] Why can R² be negative?
- [ ] Meaning of R² = 1
- [ ] Meaning of R² = 0
- [ ] Why R² is not ordinary accuracy

### Adjusted R²

- [ ] Formula
- [ ] What is n?
- [ ] What is k?
- [ ] Why does Adjusted R² exist?
- [ ] Why can Adjusted R² decrease?
- [ ] When can Adjusted R² increase?

---

# 53. What We Have Completed

We have completed the learning and implementation block for:

```text
Simple Linear Regression
```

with the following workflow:

```text
Dataset
   ↓
Pandas exploration
   ↓
Statistics
   ↓
Visualization
   ↓
Feature/target selection
   ↓
Train/test split
   ↓
Model training
   ↓
Prediction
   ↓
Actual vs predicted comparison
   ↓
Error/residuals
   ↓
MAE
   ↓
MSE
   ↓
RMSE
   ↓
R²
   ↓
Adjusted R²
```

### Important actual results from our model

```text
R² ≈ 0.773098
Adjusted R² ≈ 0.767127
```

The R² interpretation is:

> The model explains approximately 77.3% of the variation in package values relative to the mean baseline.

The Adjusted R² is slightly lower because it accounts for model predictors.

---

# 54. Next Step: Real-World Project

We should **not stop at the CGPA → Package dataset**.

The next stage is to prove that we can apply Linear Regression to a different, real-world dataset.

The workflow will remain the same:

```text
1. Find a meaningful real-world regression problem
2. Understand the dataset
3. Define the business/problem question
4. Explore the data
5. Clean the data
6. Perform EDA
7. Select features and target
8. Visualize relationships
9. Split into train/test
10. Train Linear Regression
11. Make predictions
12. Evaluate with MAE
13. Evaluate with MSE
14. Evaluate with RMSE
15. Evaluate with R²
16. Consider Adjusted R² where appropriate
17. Interpret results
18. Analyze limitations
19. Improve the model if appropriate
20. Document everything
21. Push the project to GitHub
```

The real-world project should **not simply copy the placement example**.

We want to demonstrate that we understand the complete process:

```text
Problem
  ↓
Data
  ↓
Reasoning
  ↓
Model
  ↓
Evaluation
  ↓
Conclusion
```

That will be much more valuable for a portfolio.

---

# Final Mental Model

If you remember only one flow, remember this:

```text
                SIMPLE LINEAR REGRESSION

                     Dataset
                        ↓
                Understand the data
                        ↓
                 Visualize X vs y
                        ↓
                 Choose X and y
                        ↓
                Train/Test Split
                        ↓
                   model.fit()
                        ↓
                 Model learns line
                        ↓
                model.predict()
                        ↓
              Actual vs Predicted
                        ↓
              Calculate Errors
                        ↓
        ┌────────┬────────┬────────┐
        ↓        ↓        ↓        ↓
       MAE      MSE      RMSE      R²
                                  ↓
                            Adjusted R²
                                  ↓
                           Interpretation
```

## The most important distinction

```text
Notebook
↓
Experiment, run code, inspect outputs, learn

README
↓
Permanent revision/reference document

GitHub
↓
Portfolio + proof of work
```

This separation lets us learn deeply without losing what we learned.

---

## Important note about scope

This README documents the **Simple Linear Regression learning block we completed**: the Pandas workflow, visualization, train/test split, model training/prediction, MAE, MSE, RMSE, R², and Adjusted R², including the small mathematical examples we worked through.

It should not be treated as the end of **all regression/ML topics**. Topics such as Multiple Linear Regression, assumptions/diagnostics, regularization, polynomial regression, feature engineering, cross-validation, and more advanced regression methods belong to later learning blocks.
