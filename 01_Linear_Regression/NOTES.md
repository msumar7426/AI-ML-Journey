Linear Regression: Concept Notes

Grounded in `learningSimpleLinearRegression.ipynb` (CGPA → Package) and `learningMultipleLinearRegression.ipynb`. Written so future-you can read this cold and reconstruct the logic without re-watching anything.

---

Simple Linear Regression

The idea: find the one straight line `y = mx + b` that fits a scatter of points best, here, predicting `package` from `cgpa`.

The from-scratch formula, in plain words:

```python
class SimpleLinearRegression:
    def fit(self, X_train, y_train):
        num = 0
        den = 0
        for i in range(X_train.shape[0]):
            num += (X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean())
            den += (X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean())
        self.m = num / den
        self.b = y_train.mean() - (self.m * X_train.mean())
```

`m` (the slope) asks: "when X moves away from its average, does y move away from its average in the same direction, and by how much relative to how much X itself typically moves?" The numerator measures X and y moving together (if both are usually above-average or both below-average at the same time, this is a big positive number). The denominator measures how spread out X is on its own. Slope = "shared movement" ÷ "X's own spread."

`b` (the intercept) is just: start from the average y, then walk back by `m` times the average X, so the line is guaranteed to pass through the point (mean of X, mean of y).

The gotcha worth remembering: after `train_test_split`, `X_train`/`y_train` keep their original shuffled row numbers as their index. Looping with `X_train[i]` for `i = 0, 1, 2...` does NOT reliably grab row `i`, pandas will try to match the shuffled label, not the position. Fix: call `.values` (or `.values.ravel()` for a 1D array) before passing data into the class, which converts to a plain NumPy array with normal 0, 1, 2... positional indexing. Nothing about the class itself needs to change, the fix belongs at the call site, not inside `fit`/`predict`.

---

Multiple Linear Regression: the Normal Equation

The idea: same goal as simple LR, but with many features at once. Instead of one slope, you get one coefficient per feature, solved for all at once using matrix algebra instead of a loop.

The formula: β̂ = (XᵀX)⁻¹Xᵀy, this is a closed-form solution, meaning it directly computes the exact best-fit coefficients in one shot via matrix math, with no iteration and no learning rate (that's the difference from gradient descent, same destination, different route).

```python
class MultipleLinearRegression:
    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)          # prepend a column of 1's
        betas = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

Why prepend a column of 1's? The intercept is really just "a feature that's always equal to 1, with its own coefficient." By sticking a column of 1's onto X, the intercept gets solved for by the exact same matrix equation as every other feature, no special-casing needed. That's why `betas[0]` comes out as the intercept and `betas[1:]` as the feature coefficients.

Verified in the notebook: `np.allclose(sklearn_lr.coef_, mul_lr.coef_)` and `np.isclose(sklearn_lr.intercept_, mul_lr.intercept_)`, the from-scratch version matches scikit-learn's, which is the actual proof the formula was implemented correctly, not just "it ran without an error."

---

Metrics: grounded in the tiny worked example

The notebook used a tiny hand-checkable example before trusting the real model:
`actual = [10, 20, 30, 40]`, `predicted = [11, 18, 31, 39]`.

MAE (Mean Absolute Error): average of `|actual - predicted|`. Plain words: "on average, how far off am I, not caring whether I overshot or undershot?"

MSE (Mean Squared Error): average of `(actual - predicted)²`. Squaring does two things: it makes every error positive (no cancellation between over- and under-shoots), and it punishes big misses much harder than small ones, an error of 10 contributes 100, an error of 2 contributes only 4.

RMSE: `√MSE`. This undoes the squaring, bringing the number back into the original units (rupees, package-in-lakhs, whatever), which is why RMSE is usually the one people actually quote instead of raw MSE.

R² (Coefficient of Determination):

```python
sst = sum((y - mean) ** 2 for y in actual)              # Total Sum of Squares
sse = sum((y - y_hat) ** 2 for y, y_hat in zip(actual, predicted))  # Sum of Squared Errors
r2 = 1 - (sse / sst)
```

Plain words: SST is "how wrong would I be if I just predicted the average every single time, ignoring the model entirely?", the dumbest possible baseline. SSE is "how wrong is my actual model?" R² = 1 - (my model's error ÷ the dumb baseline's error). If R² = 0.9, the model explains 90% of the variation that the dumb "always guess the average" baseline couldn't. R² = 0 means the model is no better than guessing the mean; R² can even go negative if the model is worse than that.

Adjusted R², why it exists:

```python
adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - 1 - k)   # n = rows, k = number of features
```

Plain-word problem it solves: plain R² can only go up (or stay flat) every time you add a feature to the model, even a completely random, useless column, because more features always give the model more freedom to fit the training data's noise. That makes R² alone a bad way to decide "should I add this feature?" Adjusted R² subtracts a penalty based on how many features (`k`) you're using relative to how many rows (`n`) you have, so it only goes up if the new feature explains more than what pure chance would predict. If R² goes up but adjusted R² goes down after adding a feature, that feature is very likely not real signal.

---

Standing rule
When gradient descent's notebook has real code in it again, this file should also cover: why gradient descent exists as an alternative to the normal equation (hint: the normal equation needs to invert `XᵀX`, which gets computationally expensive/unstable with many features, gradient descent scales better), the update rule, and the role of the learning rate.
