# ML Project Workflow Template

> Use this every time you start a real project, before writing the first line of code.
> Companion to `AI_ML_Journey_Guidelines.md` (that file governs how you learn a CampusX topic; this one governs how you execute a project once you're building something real).

---

## Stage 0: Problem Framing
First instinct: don't open the dataset yet. Answer these in a markdown cell first.

- What exactly am I predicting? (name the target column)
- Regression or classification?
- Who/what would use this, and how would they use a wrong prediction badly?
- What does "good" look like, which metric reflects that, and why that one over the others?
- What's the scope of this data? (what can this model NOT be trusted to generalize to, e.g. a model trained on Karachi listings has no business predicting Lahore prices without checking first)

## Stage 1: Data Loading and Structural Inspection
First instinct: check the data is mechanically sound. Do not analyze meaning yet.

- Load it. Check `.shape`, `.dtypes`, `.info()`, `.describe()`.
- Are dtypes what they should be? (a number stored as text with units, like `"156,000 km"`, is a structural problem, not an analysis question)
- Missing values? Duplicates?

## Stage 2: Data Cleaning
First instinct: make the data trustworthy before asking anything of it. This never invents new information, it only fixes what's broken.

- Duplicates: exact duplicates AND domain-specific near-duplicates (two rows can look identical on the columns that matter without being byte-identical)
- Wrong dtypes → convert (strip units, strip commas, cast to numeric)
- Missing / placeholder values (a `0` that really means "unknown" is a placeholder, not a real zero)
- Invalid / impossible values (negative mileage, a manufacture year in the future)

## Stage 3: EDA (Exploratory Data Analysis)
First instinct: ask questions of the cleaned data. You are understanding it, not changing it.

- Univariate: one column at a time, distribution shape, skew, outliers (histograms, boxplots)
- Bivariate: each feature against the target, does mileage actually relate to price the way you'd expect? Any surprises?
- Multivariate: features against each other, correlation heatmap, multicollinearity risk (matters a lot for multiple linear regression specifically)

EDA produces insight, not code that changes the dataframe. If a cell modifies `df`, it's not EDA anymore, it's cleaning or feature engineering.

## Stage 4: Feature Engineering
First instinct: now that you know what the data looks like (Stage 3), create or transform variables to expose that signal to the model.

- Derived features from domain knowledge (e.g. `age` from `year`)
- Transform skewed variables (e.g. `log(price)` if price is heavily right-skewed)
- Bucket/bin a feature if its relationship to the target is non-linear in a way linear regression can't capture directly
- Anything specific to this domain that a generic tutorial wouldn't tell you (used cars: brand tier, engine size bucket, etc.)

## Stage 5: Preprocessing for Modeling
First instinct: prepare the final numeric matrix the algorithm actually needs. This is mechanical, not insight-driven.

- Encode categoricals (one-hot / ordinal, pick based on whether there's a real order)
- Scale numeric features if the algorithm needs it (gradient-descent-based linear regression: yes; sklearn's closed-form `LinearRegression`: not strictly required, but good habit)
- Train/test split, before fitting any encoder/scaler, so nothing about the test set leaks into training

## Stage 6: Modeling
- Always build a dumb baseline first (e.g. predict the mean price), your real model must beat it, or something's wrong
- Then the actual model for the topic you're on (right now: multiple linear regression)

## Stage 7: Evaluation and Iteration
- MAE / RMSE / R² / Adjusted R², on the test set, not train
- Look at residuals, not just the headline metric
- Name what you'd try next and why

---

## The confusion this template exists to prevent

Cleaning fixes what's broken (wrong type, duplicate, impossible value), it doesn't create anything new, it makes existing data correct.

EDA asks questions and produces understanding (a chart, a correlation number, an observation), it never modifies the dataframe.

Feature engineering creates or transforms variables based on what EDA taught you, it changes the dataframe, but every change should trace back to a reason you found in Stage 3, not a habit copied from a tutorial.

Preprocessing is the mechanical last mile (encoding, scaling, splitting) that has nothing to do with insight, it's just what the math of the algorithm requires to run at all.

If you can't say which stage a line of code belongs to, that's the sign to stop and ask.
