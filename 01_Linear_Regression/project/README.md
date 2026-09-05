# Used Car Price Prediction (Karachi Market)

A multiple linear regression model that predicts the asking price of used cars listed in Karachi, Pakistan, built from a real 15,171 row scrape of a Karachi car marketplace.

## Problem

Given a used car's year, mileage, engine size, fuel type, transmission, manufacturer, and variant, predict its listing price in Pakistani Rupees (PKR).

This is a regression problem aimed at a prospective buyer trying to judge whether an asking price is fair. The model is trained on **listed** prices, not confirmed sale prices, and on Karachi data only, so it should not be assumed to generalize to other cities or to a different time period without retraining. See Stage 0 in the notebook for the full framing.

## Dataset

- Source: real listings scraped from a Karachi used car marketplace, dated December 13, 2025.
- Size: 15,171 rows, 10 columns as scraped (year, manufacturer, fuelType, name, transmission, engine, mileage, listingPrice, variant, age).
- File: `used_car_listings_13_12_2025.csv`

## Approach

The project follows an 8 stage workflow (see `templates/` at the repository root for the general template):

1. **Problem Framing.** Define the target, the metric, and the scope the model should not be trusted beyond.
2. **Data Loading and Structural Inspection.** Check shape, types, exact and near duplicates.
3. **Data Cleaning.** Remove duplicates, fix data types (mileage stored as text with commas), drop columns that are redundant with others (`age` versus `year`, `name` versus manufacturer/variant/year).
4. **Exploratory Data Analysis.** Univariate and bivariate analysis of every feature against price, using histograms, scatter plots, and boxplots. Surfaced several data quality issues fixed back in Stage 2 (placeholder mileage and engine values at suspicious round numbers, inconsistent fuel type casing).
5. **Feature Engineering.** Log transform on `listingPrice` to correct right skew. Top N plus Other grouping on high cardinality categorical columns (`manufacturer`, `variant`) to avoid a very sparse one hot encoding.
6. **Preprocessing.** One hot encoding with `drop_first=True`, an 80/20 train/test split, and feature scaling fit on training data only.
7. **Modeling.** Three models trained and compared: a baseline that always predicts the mean, scikit-learn's `LinearRegression`, and a from scratch implementation of multiple linear regression solved directly with the Normal Equation.
8. **Evaluation.** MAE, MSE, RMSE, R2, and Adjusted R2 on a held out test set, plus a residual plot and a rupee denominated error check.

The full reasoning behind every decision, written for a beginner audience with worked examples, lives in `notes/NOTES.md`.

## Results

| Model | MAE (log scale) | RMSE (log scale) | R2 | Adjusted R2 |
|---|---|---|---|---|
| Baseline (predict the mean) | 0.7561 | 0.9745 | -0.0001 | - |
| scikit-learn LinearRegression | 0.1742 | 0.2711 | 0.9226 | 0.9190 |
| Custom Normal Equation implementation | 0.1742 | 0.2711 | 0.9226 | 0.9190 |

The model explains about 92 percent of the variance in log listing price on data it has never seen, a large improvement over the baseline. The custom from scratch implementation matches scikit-learn's results to four decimal places, confirming the Normal Equation was implemented correctly. See the notebook's final cells for the same error expressed directly in rupees, and for a residual plot checking for systematic bias.

## Repository Structure

```text
project/
  used_car_price_prediction.ipynb     the full analysis, stage by stage
  used_car_listings_13_12_2025.csv    the raw dataset
  notes/
    NOTES.md                         full reasoning, explained in plain language, for every stage
```

## Tools Used

Python, Pandas, NumPy, Matplotlib, scikit-learn, Jupyter Notebook.

## How to Run

```bash
pip install -r ../../requirements.txt
jupyter notebook used_car_price_prediction.ipynb
```

Run all cells in order from top to bottom.

## Limitations

- Trained on Karachi listings only, for one point in time (December 2025). Prices are expected to drift with currency and fuel price changes, and the model has not seen data from any other city.
- Predicts asking price, not confirmed transaction price, in a market where negotiation is common.
- The `variant_grouped` feature has many categories (72), some backed by only a couple dozen rows, so coefficients for rare variants are noisier than for common ones.
- Only main effects are modeled; no interaction terms (for example, mileage's effect on price differing by fuel type) were explored, since plain multiple linear regression does not use them without being explicitly engineered.

## Author

M S Umar, Computer Science student, learning AI and Machine Learning.

- GitHub: https://github.com/msumar7426
- Email: muhammadsiddiqueumar1@gmail.com
