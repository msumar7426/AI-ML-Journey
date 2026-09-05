# Used Car Price Prediction: Concept Notes

Written to be read cold, by anyone, not just future-me. If you're a beginner reading someone else's GitHub repo, this is meant to walk you through not just what the code does, but how we figured out what to do and why, mistakes included. General ML/stats theory (linear regression, metrics) lives in `../../notes/NOTES.md`, this file is specific to real decisions made on this exact dataset.

Dataset: 15,171 real used-car listings scraped from a Karachi marketplace (`used_car_listings_13_12_2025.csv`). Target: `listingPrice`.

---

## Part 1: Calling a method vs. just naming it

Before touching duplicates, this tripped us up twice, so it's worth its own section.

```python
def greet():
    return "hello"

greet       # <function greet at 0x...>  , just points at the function, doesn't run it
greet()     # "hello"                     , actually runs it
```

A function or method name on its own is just a label pointing at code sitting in memory. The `()` is the instruction "run this now." This bit us with `df.describe` (no parentheses), it didn't error, it just printed a description of the method object, which happened to include the entire dataframe as part of that description, so it looked like it worked but wasn't actually the statistics table. It bit us again with `df.duplicated.sum()`, except that one raised a real `AttributeError`, because a method object has no `.sum()`, only real data (like a Series) does. Same root bug, two different symptoms. Lesson: if output looks unexpectedly large or generic, check you actually called the method.

---

## Part 2: Finding exact duplicates

```python
df.duplicated().sum()
```

`df.duplicated()` returns a `True`/`False` Series, one value per row, `True` if that row is a repeat of an earlier one. Since Python treats `True` as `1` and `False` as `0`, summing that Series just counts how many rows got flagged.

Result: 2,477 exact duplicate rows, about 16% of the entire dataset. Worth sitting with that number before doing anything about it; that's not noise, that's one row in six.

### The `keep` parameter

`duplicated()` always spares exactly one occurrence per repeated group and flags the rest as `True`, `keep` controls which occurrence is spared:

| values in order | `keep='first'` (default) | `keep='last'` | `keep=False` |
|---|---|---|---|
| `A, A, B, A, C` (rows 0-4) | row 0 spared; rows 1, 3 → `True` | row 3 spared; rows 0, 1 → `True` | nobody spared; rows 0, 1, 3 → `True` |

For counting duplicates, the default (`keep='first'`) is fine, you just want "how many extra copies exist." For looking at a duplicate group with your own eyes, you need `keep=False`, otherwise the very first (and "spared") copy gets filtered out of your view, and you're left staring at repeats with nothing to compare them against.

### Looking at the actual rows

```python
df[df.duplicated(keep=False)].sort_values(by=['manufacturer','variant','year','mileage']).head(20)
```

Read this inside-out: `df.duplicated(keep=False)` makes the `True`/`False` map → `df[ ... ]` filters `df` down to only the flagged rows → `.sort_values(by=[...])` reorders them so matching rows land next to each other instead of being scattered by original row position → `.head(20)` caps the output to a manageable sample.

This is a reusable pattern, not just a duplicates thing:
```python
df[ <condition that picks the rows you care about> ].sort_values(by=[ <columns that cluster related rows> ]).head(N)
```

### What we found, and how we figured out what it meant

Every duplicate pair we looked at had identical mileage, down to the exact kilometer (e.g. two separate rows both reading `45,812 km`), and the two copies of each pair sat far apart in the original row order (sometimes thousands of rows apart).

Two clues, one conclusion: mileage is a continuous, highly specific number, two genuinely different cars, sold by two different people, coincidentally sharing the exact same odometer reading is essentially implausible. And if the duplication came from something like the scraper hitting the same page twice in a row, you'd expect the copies to sit close together in row order, not scattered across the whole file. Far-apart row positions plus an exact match on a high-cardinality number like mileage points at one thing: the same listing got captured more than once during separate scraping passes, a data-collection artifact, not two real cars that happen to be identical.

---

## Part 3: Finding near-duplicates

Exact-duplicate detection only catches rows where every single column matches. But what about the same physical car, listed once, then listed again months later after being driven a bit more, mileage and maybe price would differ, so `duplicated()` would never flag it. That needs a looser check.

```python
df[df.duplicated(subset=['manufacturer','variant','year','engine','fuelType','transmission'], keep=False)] \
    .sort_values(by=['manufacturer','variant','year','engine','fuelType','transmission']).head(20)
```

`subset=[...]` tells `duplicated()` to only compare the listed columns, everything else is ignored. Deliberately left out of the subset: `mileage` and `listingPrice`, those are exactly the two things we expect might legitimately differ between two snapshots of the same real car, and if we required them to match too, we'd miss the pattern we're trying to catch.

### Two very different things came back, and they need different handling

Audi A3 2016 (3 rows, all `Petrol`/`Automatic`/`1200cc`): mileages `101,000` / `41,000` / `62,000 km`, prices `5,950,000` / `6,700,000` / `5,850,000`, no two agree on anything. Conclusion: this is not duplication. Three different sellers, three different actual cars, that happen to share a common configuration. Real, legitimate, distinct data, must not be dropped.

Audi A4 2014 (part of the group): an already-known exact-duplicate pair (`48,000 km`, `6,500,000`, twice) plus one more row at `48,810 km` and the same `6,500,000` price. Off by 810 km, identical price. That pattern, a small, plausible, real-world mileage increase with an unchanged asking price, is a much better match for "the same car, seen again months later" than for a scraper bug. (A bug would more likely reproduce numbers exactly, the way the true exact-duplicate pair already does, not invent a believable 810 km bump on its own.)

Takeaway: "near-duplicate" isn't one category. It splits into at least two: coincidental matches on a popular configuration (keep, they're real), and genuine same-car-at-different-times matches (probably redundant, worth merging).

### The cleaning decision, and why "good enough, documented" was the right call here

We could try to build a fully general "is this really the same car" classifier. For a first project, that's disproportionate, this dataset has no seller ID or listing ID to confirm identity with certainty, so perfect confidence isn't available at any amount of effort, only diminishing returns.

The rule we're using instead: a near-duplicate group counts as "same car" only if price matches exactly and mileage is within a small gap (2,000 km, comfortably above the genuine case we found at 810 km, comfortably below the coincidence case's tens-of-thousands-km gaps). Anything that doesn't clear that bar is left alone as legitimate distinct data.

This is a deliberate, stated simplification, not a hidden shortcut, and that distinction matters for a portfolio project. "I used a documented mileage-gap heuristic, and here's its known limitation" reads as engineering judgment. Silently pretending the cleaning is perfect, or burning days chasing certainty the data can't provide, both read worse.

---

## Part 4: Converting text columns into real numbers

Before this step: `df_clean = df.drop_duplicates(keep='first')` dropped the 2,477 exact duplicates (verified: `df_clean.shape` → `(12694, 10)`). But `mileage` (`"156,000 km"`) and `engine` (`"1800cc"`) are still text, not numbers, and no math (comparisons, model training, `groupby` aggregation) works correctly on text that merely looks like a number. Rule of habit going forward: before transforming or comparing any column, check `.dtype`, if it says `object`/`str` but the values look numeric, that's a sign it needs converting first.

### Mileage

```python
df_clean['mileage'] = df_clean['mileage'].str.replace(',', '').str.replace(' km', '').astype(int)
```

`.str` unlocks string operations across an entire column at once, instead of writing a loop. `.str.replace(',', '')` deletes every comma; `.str.replace(' km', '')` deletes the unit suffix; `.astype(int)` converts the now-clean text (`"156000"`) into an actual integer (`156000`).

### Engine: the same idea, plus a real landmine

The naive version, `df_clean['engine'].str.replace('cc', '').astype(int)`, crashes. Here's why, and how we found it.

Finding it: `.head(20)` only shows the first 20 rows in file order, it will not reliably surface a rare pattern (1.5% of rows, scattered throughout the file). The actual tool for this is `.nunique()` then `.unique()`:

```python
df['engine'].nunique()   # 85, small enough to read every one
df['engine'].unique()    # prints all 85 distinct values, including the odd one out
```

With only 85 distinct values, it's easy to eyeball every one and spot `'cc'` sitting alone with no digits in front of it, 224 rows, and every single one of them turned out to be `fuelType == 'Electric'`. Electric cars have no combustion engine, so there's no displacement number to report; the site just left the label with nothing in it. Not corrupted data, a real, meaningful gap.

For a column too large to eyeball (`mileage` has 2,749 distinct values), the better tool is a regex mismatch filter, flip the question from "show me every value" to "show me only the ones that break the pattern I expect":

```python
df[~df['mileage'].str.match(r'^[\d,]+ km$')]   # rows that do NOT look like "<digits/commas> km"
```
(Ran this on `mileage`, zero mismatches, so that column was already safe.)

The decision: for the 224 electric-car rows, set `engine = 0` rather than leaving it missing. Reasoning: `fuelType` is already a separate column that will go into the model, a plain multiple linear regression is additive, so the `engine` coefficient explains price-per-cc for combustion cars, while the `fuelType_Electric` indicator separately explains the "being electric" baseline shift. Setting `engine = 0` contributes nothing extra from the engine term for those rows, so it doesn't distort the combustion-car relationship. `NaN` would just force the same 0-vs-something decision later, since regression can't train on missing values anyway. One thing to remember: a future univariate plot of `engine` alone will show a spike of 224 cars at 0, that's expected, not a bug.

### The actual fix

```python
df_clean['engine'] = df_clean['engine'].str.replace('cc', '')
df_clean.loc[df_clean['engine'] == '', 'engine'] = '0'
df_clean['engine'] = df_clean['engine'].astype(int)
```

### Traced through one row of each kind

| | after line 1 (`str.replace('cc','')`) | after line 2 (`.loc[...]='0'`) | after line 3 (`.astype(int)`) |
|---|---|---|---|
| Toyota, was `"1800cc"` | `"1800"` | unchanged, condition is `False` for this row | `1800` |
| Nissan Leaf, was `"cc"` | `""` (empty) | condition is `True` → becomes `"0"` | `0` |

`.loc[<condition>, '<column>'] = <value>` is the general recipe for "find exactly these rows, and only in this one column, write this value", different from `df[condition]`, which only lets you look, not change anything. This is a pattern worth reusing any time a small subset of rows needs fixing rather than the whole column.

A subtlety caught along the way: an earlier attempt used `.str.replace('cc', ' ')` (a space) instead of `''` (nothing). It happened to still work, because Python's `int()` quietly strips whitespace, but that's the code working by accident, not by design. Worth remembering: "it ran without an error" isn't the same as "it did what I meant."

---

## Part 5: Finishing Stage 2: near-duplicates, and dropping `age`

Investigating near-duplicate groups. Grouped by every spec that should make two rows "the same car" (`manufacturer`, `variant`, `year`, `engine`, `fuelType`, `transmission`) and looked at the `mileage` spread inside each group:

```python
summary = df_clean.groupby(['manufacturer','variant','year','engine','fuelType','transmission'])['mileage'].agg(['count','min','max'])
summary['range'] = summary['max'] - summary['min']
multi = summary[summary['count'] > 1]   # groups with more than one row
```

`count` alone isn't proof of duplication, a popular model can genuinely have many different, low-mileage listings. `range` (max mileage - min mileage) is a better signal for a tight cluster. But it has a real limitation: for a group of 3+ rows, one genuinely-different car can inflate the range enough to make a real duplicate pair invisible. Example found in the data, Audi A4 2014 had mileages `48,000 / 48,810 / 52,155`. The first two are almost certainly the same car re-listed; the third is a different car. But the group's overall range is `4,155`, which would fail a strict "range ≤ 2,000" threshold and wrongly throw out the whole group, real pair included.

The decision. Rather than build a more precise per-pair comparison (worth doing eventually, not for a first project), the rule used here treats two rows as the same car only if they match on every spec and `listingPrice` exactly:

```python
df_final = df_clean.drop_duplicates(
    subset=['manufacturer','variant','year','engine','fuelType','transmission','listingPrice'],
    keep='last'
)
```

Documented limitation: this doesn't separately check mileage closeness, so in rare cases it could merge two different cars that coincidentally share every spec and the exact same asking price, or fail to merge a genuine re-listing whose price changed. Good enough for a first project, worth revisiting with a sharper (e.g. per-pair, sorted-mileage-gap) check later.

Dropping `age`. Confirmed earlier via `.describe()`: `age` and `year` had identical standard deviations to six decimal places, proof `age` is just `constant - year`, i.e. the exact same information written a different way (perfect multicollinearity). Keeping both would double-count that information for the model, so `age` is dropped and `year` is kept (more directly useful, and doesn't go stale the way "age as of when this was scraped" would):

```python
df_final = df_final.drop(columns=['age'])
```

From here on, `df_final` is the dataframe Stage 3 (EDA) and everything after it uses, not `df_clean`.

---

## Part 6: Stage 3 EDA, Univariate: listingPrice, mileage, and a second cleaning pass

Univariate means looking at one column completely on its own, no comparisons to anything else yet, not even the target. The goal is to understand shape, spread, and outliers before asking how columns relate to each other.

### listingPrice

Plotted with `df_final['listingPrice'].hist(bins=50)`. The prediction going in, based on Stage 1's mean (5.27M) being well above the median (3.2M), was a big cluster of ordinary-priced cars on the left with a long, thin tail of expensive ones stretching right. That is exactly what the histogram showed: a tall bar of roughly 5,600 cars priced under about 3 million PKR, then a rapid drop-off, with the handful of cars priced 20 million and up flattening out almost invisibly at this scale. This is why the mean sits so far above the median: the median only cares how many values are on each side, the mean is pulled upward by the size of the extreme few, even though there are barely any of them.

### mileage

Plotted the same way. This one did not match a simple single-peak shape. There are two humps: a bar near 0 km, then a dip around 20,000 to 40,000 km, then an even bigger peak around 80,000 to 100,000 km, before a long tail out toward very high mileage. Working theory for the dip: cars get listed either very early (someone decides quickly the car is not right for them, or upgrades soon after buying) or after a typical few years of ownership once mileage has built up to a "normal used car" level, but relatively few people list a car in between, once someone has driven a car for a year or two without issue, there is less reason to sell it yet. This cannot be proven without ownership-duration data, which this dataset does not have, so it is recorded here as a plausible explanation, not a confirmed one.

### A placeholder problem found while looking at mileage's outliers

Sorting `df_final` by mileage descending (`.sort_values(by='mileage', ascending=False).head(30)`) surfaced two distinct problems in the tail:

A tight cluster of suspiciously identical or near-identical values sitting right at a ceiling: 13 rows at exactly `1000000`, plus 4 at `999999`, and one each at `999998`, `999990`, and `999000`. These appear on completely unrelated cars, a 1989 Suzuki Mehran, a 1980 Mazda Titan, a 2005 Honda Civic, different ages, brands, and prices, all sharing almost the same "mileage." Real odometer readings do not coincidentally cluster at a round number like that across unrelated cars. This is a placeholder, the same shape of problem as the electric cars' bare `'cc'` engine value from Part 4, something standing in for "unknown" rather than a real reading, most likely something the scraped site defaults to when the real mileage was missing or unreadable.

Beyond that cluster, some remaining high values were still implausible once checked against the car's age. A 2018 Suzuki Swift listed at 980,000 km implies roughly 140,000 km driven per year, which is not realistic for a personal car. Meanwhile a 1984 Daihatsu Charade at 991,555 km (roughly 24,000 km per year over 41 years) and a 1988 Honda Accord at 987,546 km looked like genuine, if high, real-world values, non-round numbers attached to decades-old cars. The distinguishing signal was not the size of the number by itself, it was whether the number was suspiciously round and whether it made sense given the car's age.

The fix, applied back in Stage 2 since it changes `df_final` (per the workflow template's own rule: EDA understands, cleaning changes):

```python
# Remove the placeholder cluster sitting right at the ceiling (1,000,000 and near-identical values)
df_final = df_final[df_final['mileage'] < 999000]

# Remove remaining rows where mileage is implausible for the car's age
reference_year = 2025
age_years = (reference_year - df_final['year']).clip(lower=1)
implied_km_per_year = df_final['mileage'] / age_years

df_final = df_final[implied_km_per_year <= 50000]
df_final.shape
```

`age_years` is how old the car is (current year minus model year), `.clip(lower=1)` avoids dividing by zero for a brand new car. `implied_km_per_year` spreads the mileage evenly across those years, a rough estimate of how much the car was driven annually. 50,000 km/year is a generous ceiling, well above normal personal use, but low enough to catch something like the Swift's implied 140,000 km/year.

Documented limitation: this is a heuristic, not a certainty. A 1980 Toyota Corolla at exactly 990,000 km passes the rate check (about 22,000 km/year over 45 years) even though the round number is still a little suspicious, we cannot fully resolve every ambiguous case, and 50,000 km/year as a cutoff is a judgment call rather than an exact law. Same philosophy as the near-duplicate rule in Part 3: a reasonable, stated simplification beats chasing certainty the data cannot provide.

---

## Part 7: Stage 3 EDA, Univariate: engine

Plotted the same way as before: `df_final['engine'].hist(bins=50)`. Two things stood out immediately. A small bar right at 0, exactly as expected from Part 4, the 224 electric cars. And two separate tall peaks rather than one, roughly 800cc and roughly 1300-1600cc, matching the two most common car segments in this market, small economy hatchbacks and mid-size sedans.

### A placeholder and a set of individual errors, found the same way as mileage

Sorting descending by engine (`.sort_values(by='engine', ascending=False)`) surfaced a mix of problems, messier than mileage's, because engine size doesn't have one universal "normal" range the way mileage-per-year does, a truck, a kei car, and a luxury sedan all have completely different normal engine sizes.

The clearest signal: three completely unrelated cars, a Suzuki Vitara (SUV), a Honda Civic (sedan), and a Toyota Land Cruiser (4x4), all shared the identical value `15000` (15 liters, impossible for any of them). Same signature as mileage's `1000000` placeholder, an identical value across unrelated car types.

Beyond that, several individual rows had engine sizes that don't make sense for what the car actually is: a Toyota IST (compact car) at `13004`, a Mitsubishi Pajero at `11000`, a Toyota Crown at `8000`, a Suzuki Alto (a tiny economy car) at `8000`, and a Daihatsu Move at `6600`. The Move is the clearest individual case: kei cars are a Japanese vehicle category with a legal engine cap of exactly 660cc, so `6600` is almost certainly the same "extra zero" mistake suspected earlier for mileage. An Isuzu NKR (a light delivery truck) at `6650` is a genuine gray area, trucks do run larger engines, so this one might be real, but it's on the high side even for that, and was not spared.

### The case that looked like an error but wasn't

A Mercedes-Benz S-Class at `6203` and seven separate Mercedes-Benz C-Class listings (spanning 2009 to 2013) at `6200` at first looked suspicious simply for being large numbers. But this is different from the placeholder cases: the value repeats consistently across the *same* model across *different* years, not across unrelated car types. That pattern matches a real shared engine spec, not a scraper artifact, and indeed the high-performance AMG variants of both the C-Class and S-Class genuinely carry engines around 6.2 liters. This was deliberately kept, not dropped.

The lesson: "this value repeats identically across multiple rows" is not, by itself, proof of a placeholder. What matters is *what* is repeating it. Unrelated cars sharing an impossible number is a red flag. The same model sharing a real-world-plausible number across different years is exactly what genuine data looks like.

### The fix

```python
# 6500 sits in the gap between the highest genuine value found (Mercedes AMG variants, ~6200-6203cc)
# and the lowest confirmed error (Daihatsu Move at 6600cc, impossible for a 660cc-capped kei car)
df_final = df_final[df_final['engine'] <= 6500]
df_final.shape
```

Applied back in Stage 2, same rule as always: EDA discovers, cleaning changes `df_final`.

Documented limitation: the Isuzu NKR at 6650cc is dropped along with the confirmed errors even though it might be a genuine truck engine, we couldn't be fully certain either way, and 6500 was chosen specifically because it's the cleanest line available between the confirmed-real Mercedes cluster and the confirmed-fake Move value, not because it's provably the correct cutoff in every case.

---

## Part 8: Stage 3 EDA, Univariate: year (and closing out Univariate)

Fixed a copy-paste bug first: the year histogram's title still said "Distribution of Engine Size" from the previous cell. Renamed it to "Distribution of Manufacturing Year".

Plotted with `bins=200` for a finer view of the shape. The bars climb steadily from the 1950s and spike hard around 2020-2023. This makes sense: newer cars are more likely to still be on the road and worth listing for resale, while older cars have mostly already been sold, scrapped, or worn out with age. Depreciation adds to this too, since newer cars hold more resale value, sellers are more motivated to list them.

### Checking the edges instead of trusting the shape

Both mileage and engine looked fine at a glance in their histograms and still turned out to hide real placeholder problems, so rather than eyeballing the year histogram's edges and calling it clean, the edges were checked directly:

```python
df_final['year'].min()   # 1951
df_final['year'].max()   # 2025
```

Neither is a placeholder. `2025` lines up exactly with when this dataset was scraped (December 2025), a listing for a current-year model is completely normal. `1951` is old but not physically impossible the way engine's 6600cc kei car was, someone can genuinely be selling a decades-old vintage car.

The key difference from the mileage and engine problems: those were flagged because the *same* impossible value kept repeating across unrelated cars, that repetition is the actual fingerprint of a placeholder. A single old year doesn't have that fingerprint. Confirmed with a count:

```python
(df_final['year'] == 1951).sum()   # 1
```

Just one row. A genuine rare vintage listing, not a data-entry artifact. No fix needed for `year`.

### Univariate, wrapped up

Every column now has a defensible, documented reason for its distribution shape and its edges: `listingPrice` (right-skewed, expected for prices), `mileage` (bimodal, cleaned of two placeholder/implausibility issues), `engine` (two-peak, cleaned of a placeholder cluster and individual errors while keeping a genuine Mercedes AMG cluster), and `year` (climbing toward recent years, both edges verified genuine).

---

## Part 9: Stage 3 EDA, Bivariate: mileage, engine, year vs listingPrice, and a boxplot detour into fuelType

### Numeric features vs price: scatter plots

Bivariate means looking at two variables together instead of one, specifically here, each feature against `listingPrice`, since price is what we are ultimately trying to predict.

For two numeric columns, the tool is a scatter plot: one dot per row, x-position from the feature, y-position from price.

```python
plt.scatter(df_final['mileage'], df_final['listingPrice'])
plt.xlabel('Mileage (km)')
plt.ylabel('Listing Price (PKR)')
plt.title('Mileage vs Listing Price')
plt.show()
```

Same pattern repeated for `engine` and `year`.

What this showed: `mileage` vs price is a steep decay curve, price drops fast at low mileage and flattens out at high mileage, not a straight line. `engine` vs price is noisy and scattered, no clean trend, expensive cars show up across many different engine sizes. `year` vs price is the clearest of the three, flat and low for decades, then curving sharply upward for recent years, the mirror image of the mileage curve, which makes sense since mileage and year are related.

### Why a curved relationship does not rule out linear regression

Seeing a curve instead of a straight line here is not a dead end. It is expected. Real-world price-vs-usage relationships (car price vs mileage, house price vs age, phone resale value vs months used) are almost always exponential-decay shaped, not linear, and this connects directly back to the Stage 1 finding that `listingPrice` itself is right-skewed. Log-transforming price (planned for Stage 4/5, Feature Engineering/Preprocessing) is the standard fix for exactly this kind of curve, and it is already on the roadmap. Linear regression does not require every raw feature to already look linear against price, it requires the relationship to become roughly linear after the right transformation.

### Categorical features vs price: boxplots

A scatter plot needs two numeric axes, so it does not work for a categorical column like `fuelType` (you cannot plot "Petrol" on a number line). The tool for categorical-vs-numeric is a boxplot instead: one box per category, showing the spread of prices within that group.

```python
df_final.boxplot(column='listingPrice', by='fuelType')
plt.xlabel('Fuel Type')
plt.ylabel('Listing Price (PKR)')
plt.title('Listing Price by Fuel Type')
plt.suptitle('')
plt.show()
```

Minor gotcha worth remembering: pandas' `.boxplot()` automatically adds its own title above the plot (something like "Boxplot grouped by fuelType"), separate from `plt.title()`. Without `plt.suptitle('')` to clear it, two titles stack on top of each other.

### Reading a boxplot

The box spans from Q1 (25th percentile) to Q3 (75th percentile), the middle 50% of prices in that group. The line inside the box is the median. The whiskers extend to the most extreme value that is still within 1.5x the box height (the IQR) from the box. Anything further out than that is drawn as an individual dot, an outlier, a car priced unusually high or low for its group.

Worked through a tiny made-up example (11 prices: 10, 12, 13, 14, 15, 15, 16, 17, 18, 20, 45) to see the mechanics: median 15, Q1 13.5, Q3 17.5, IQR 4, so the "normal" range extends to roughly 1.5x4=6 beyond the box (7.5 to 23.5). The value 45 falls far outside that range, so it gets drawn as an outlier dot rather than stretching the whisker up to it.

Reading the actual `fuelType` plot: Petrol has a huge column of outlier dots since it is by far the most common fuel type, spanning everything from cheap economy cars to expensive luxury ones. Electric has a noticeably higher median and wider box than Petrol, Diesel, or Hybrid. CNG, LPG, and PHEV are compressed near the bottom, either tightly clustered prices or few rows in that category.

### A second, smaller placeholder-style problem: fuelType casing

`df_final['fuelType'].unique()` returned 8 categories instead of the expected 7: `LPG` and `Lpg` are the same real fuel type, split into two by inconsistent capitalization. Every other value (`CNG`, `PHEV`, `Petrol`, `Diesel`, `Hybrid`, `Electric`) was already consistently formatted.

Considered blanket-uppercasing the whole column with `.str.upper()`, this would technically merge `LPG`/`Lpg` fine (pandas treats identical strings as the same category), but it would also turn the already-correct word-case values into shouting case (`Petrol` to `PETROL`, and so on). Since only one value was actually broken, fixed it directly instead:

```python
df_final['fuelType'] = df_final['fuelType'].replace('Lpg', 'LPG')
```

Same stage-boundary rule as always applied here: this was discovered while doing Bivariate EDA (Stage 3), but since it changes `df_final`, the fix itself lives back in Stage 2, not in the EDA cell that found it.

---

## Part 10: Stage 3 EDA, Bivariate: transmission

Same boxplot approach used for `fuelType`, applied to `transmission`:

```python
df_final.boxplot(column='listingPrice', by='transmission')
plt.xlabel('Transmission')
plt.ylabel('Listed Price')
plt.title('Listing Price by Transmission Type')
plt.suptitle('')
plt.show()
```

### Reading the boxplot, then verifying with real numbers

Visually, Automatic's box sat noticeably higher than Manual's, and Automatic had a huge column of outlier dots stretching up to very high prices while Manual barely had any. Read purely off the picture, that could mean two different things: Automatic cars are priced higher, and/or there are simply a lot more Automatic listings in the dataset (more rows naturally means more chances for outlier dots to appear, regardless of price patterns).

Same discipline as always: a visual impression is a starting point, not a conclusion, so it got checked against actual numbers rather than trusted as-is.

```python
df_final['transmission'].value_counts()
```

Result: `Automatic: 6611`, `Manual: 3553`, a real 65/35 split. Automatic is genuinely the majority, but not the "enormous" imbalance the crowded outlier dots first suggested.

```python
df_final.groupby('transmission')['listingPrice'].median()
```

Result: `Automatic: 4,000,000`, `Manual: 1,090,000`, roughly a 3.7x difference. This confirms the boxplot's visual pattern with a concrete, citable number instead of "the box looked higher."

### How groupby().median() works

`df_final.groupby('transmission')` splits the whole dataframe into separate buckets by the values in the `transmission` column, every `Automatic` row into one bucket, every `Manual` row into another, nothing calculated yet, just sorted. `['listingPrice']` then narrows each bucket down to just the price column. `.median()` computes the median separately within each bucket. The result is one median value per group, the same statistic the boxplot's line-inside-the-box was already showing visually, just requested directly as a number.

### Outliers are not automatically errors

Important distinction from the mileage/engine investigations: those outliers were genuine data-quality problems, physically impossible values that needed removing. Here, an outlier dot above Automatic's box is very likely a real, legitimately expensive car (a luxury automatic vehicle), not a data error. "Outlier" in a boxplot only means "statistically unusual relative to its own group," it does not by itself mean wrong, fake, or mispriced. Whether an outlier is a problem or a genuine data point always depends on domain context, not the label alone.

Conclusion: `transmission` looks like a genuinely useful predictive signal, a ~3.7x median price gap between groups is a real, sizeable difference.

---

## Part 11: Stage 3 EDA, Bivariate: manufacturer, variant, and dropping the redundant name column

### Checking cardinality before deciding anything

Before deciding how to handle `manufacturer` and `variant`, their actual cardinality was checked directly rather than assumed:

```python
df_final['manufacturer'].nunique()   # 58
df_final['variant'].nunique()        # 349
```

`print(df_final['manufacturer'].value_counts())` showed a heavily skewed distribution: Toyota (3002), Suzuki (2871), Honda (1563), and Daihatsu (886) alone make up roughly 85% of all rows, dropping off fast after that into a long tail, over a dozen brands (GMC, SsangYong, Jaguar, Willys, Hummer, and others) appear exactly once each.

`print(df_final['variant'].value_counts())` showed the same shape: Corolla (925), Alto (737), Civic (726), Cultus (567), City (468) dominate, with many variants down at a count of 1 in the tail (CR-V, Cressida, Charmant, Pixis Mega, and a stray " Mini Bus" with a leading space).

This is a classic case for a top-N-plus-Other grouping strategy once we reach Stage 4 (Feature Engineering): keep the well-represented categories as their own groups, since brand and model both genuinely predict price, and bucket the long tail of rare, single-listing categories together so they don't each get a dummy column the model can never generalize from. The actual N cutoff and implementation is deferred to Stage 4, this Stage 3 pass was only about understanding the shape of the problem.

### Discovering name is fully redundant

While working through this, a third categorical column, `name`, turned out to need a completely different decision. `df_final['name'].nunique()` returned 1749, unusually high. Looking at a sample (`df_final['name'].head(10)`) showed a clear pattern: every value is just `"{manufacturer} {variant} {year} for sale in Karachi"`, for example `"Toyota Prius 2013 for sale in Karachi"`. The `"for sale in Karachi"` suffix carries zero information, since the entire dataset is Karachi listings.

Verified this rather than trusting a 10-row sample:

```python
reconstructed = df_final['manufacturer'] + ' ' + df_final['variant'] + ' ' + df_final['year'].astype(int).astype(str) + ' for sale in Karachi'
(reconstructed == df_final['name']).sum()
```

10,050 rows matched exactly. The roughly 200 that did not were inspected directly rather than dismissed, and turned out to share one specific cause: for Mercedes-Benz rows, `variant` already includes the word "Benz" (e.g. `variant` = `"Benz C Class"` with `manufacturer` = `"Mercedes Benz"`), so the naive reconstruction doubled it (`"Mercedes Benz Benz C Class..."`) while the real `name` field only says it once. Not new information hiding in the mismatches, just an artifact of the reconstruction logic.

### The fix

Same situation as `age` vs `year` earlier in Stage 2: `name` is a fixed transform of columns that already exist (`manufacturer`, `variant`, `year`), so it contributes nothing a model doesn't already have access to. Dropped entirely rather than treated as a genuine high-cardinality feature to encode:

```python
df_final = df_final.drop(columns=['name'])
df_final.shape
```

Placed back in Stage 2, next to the `age`/`year` redundancy fix, following the same discovered-in-Stage-3-fixed-in-Stage-2 rule used throughout this project.

### A wrinkle in the discover-in-Stage-3-fix-in-Stage-2 rule

Running the notebook top to bottom after this change surfaced a real bug: `KeyError: 'name'` on a cell further down that still referenced `df_final['name']`. The reason: for mileage, engine, and fuelType, the Stage 2 fix only changes rows or values, the column itself still exists afterward, so Stage 3 cells that reference it keep working fine. Dropping a column entirely is different, once `name` is dropped in Stage 2, nothing after that point in the notebook can reference it anymore, including the very cells that originally discovered the problem (the `.nunique()`, `.head()`, and reconstruction-check cells).

Fix: those discovery cells (`name.nunique()`, `name.head(10)`, the `reconstructed` comparison, and the mismatches inspection) were moved out of Stage 3 entirely and placed in Stage 2, directly before the drop, so the notebook tells the full story, evidence then decision, in one place, and nothing downstream references a column that no longer exists. Stage 3 keeps only the short wrap-up markdown pointing back to Stage 2. Lesson for next time: a column-drop fix needs its supporting evidence moved with it, not just the drop line itself.

---

## Part 12: Stage 4, Feature Engineering

Multivariate analysis was consciously skipped to keep the project on schedule. The key relationships were already understood from Bivariate; a correlation heatmap would have been nice-to-have, not essential to a first working model.

### Log-transforming listingPrice

Traces back to Stage 1 (right-skew, mean well above median) and Stage 3 Bivariate (every scatter plot against price came out curved rather than straight). A log transform compresses large values more aggressively than small ones, treating equal ratios as equal distances, which pulls in the long right tail and should straighten those curved relationships.

```python
import numpy as np
df_final['log_listingPrice'] = np.log(df_final['listingPrice'])
```

Kept as a new column rather than overwriting `listingPrice`, so real currency values stay available for interpreting results later. The resulting histogram came out close to a symmetric bell shape, confirming the fix worked.

### Grouping manufacturer and variant into top-N-plus-Other

Traces back to Stage 3 Bivariate, where `manufacturer` (58 categories) and `variant` (349 categories) were both heavily skewed toward a handful of common values with a long tail of rare, single-listing ones. One-hot encoding either column as-is would create dozens or hundreds of columns the model could never generalize from.

```python
manufacturer_counts = df_final['manufacturer'].value_counts()
keep_manufacturers = manufacturer_counts[manufacturer_counts >= 100].index
df_final['manufacturer_grouped'] = df_final['manufacturer'].where(
    df_final['manufacturer'].isin(keep_manufacturers), 'Other'
)

variant_counts = df_final['variant'].value_counts()
keep_variants = variant_counts[variant_counts >= 20].index
df_final['variant_grouped'] = df_final['variant'].where(
    df_final['variant'].isin(keep_variants), 'Other'
)
```

`.where(condition, fallback)` keeps the original value wherever the condition is True and replaces it with the fallback wherever it's False, the opposite instinct of a normal if-statement, worth remembering since it reads backwards at first.

`manufacturer_grouped` came out to 11 categories total (down from 58): Toyota, Suzuki, Honda, Daihatsu, Nissan, KIA, Hyundai, Mitsubishi, Changan, Mercedes Benz, each kept for having 100+ listings, and `Other` holding the remaining 631 rows. `variant_grouped` used a lower threshold (20+ listings) since variants are more fragmented to begin with. Both original columns were kept alongside the grouped ones for reference.

---

## Part 13: Stage 5, Preprocessing

### The feature set

Target: `log_listingPrice`. Features: `year`, `mileage`, `engine` (numeric, unchanged), plus `fuelType`, `transmission`, `manufacturer_grouped`, `variant_grouped` (categorical). Raw `manufacturer` and `variant` are excluded, the grouped versions already carry the same signal without the unusable long tail of rare categories.

```python
feature_cols = ['year', 'mileage', 'engine', 'fuelType', 'transmission', 'manufacturer_grouped', 'variant_grouped']
X = df_final[feature_cols].copy()
y = df_final['log_listingPrice'].copy()

X = pd.get_dummies(X, columns=['fuelType', 'transmission', 'manufacturer_grouped', 'variant_grouped'], drop_first=True)
X.shape   # (10164, 91)
```

### In plain words, using four made-up cars

Car A: 2020, 10,000km, 1300cc, Petrol. Car B: 2015, 60,000km, 1500cc, Diesel. Car C: 2010, 120,000km, 800cc, Petrol. Car D: 2022, 5,000km, 1800cc, Hybrid.

`X` is the set of clues handed to the model, `y` is the answer key (the real price) being predicted, kept completely separate so the model never gets to peek at the answer while guessing.

One-hot encoding turns words into yes/no columns since a computer can't do math on the word "Petrol". With `drop_first=True`, one fuel type is picked as the silent default (say Petrol) and only gets columns for the others:

| Car | is_Diesel | is_Hybrid |
|---|---|---|
| A | 0 | 0 |
| B | 1 | 0 |
| C | 0 | 0 |
| D | 0 | 1 |

A and C both show `0, 0`, meaning Petrol without needing their own column, since if a row isn't Diesel and isn't Hybrid, it must be Petrol. Keeping a Petrol column too would just repeat information already implied by the other two, and that redundancy (called the dummy variable trap) confuses linear regression's math.

### Train/test split, then scaling, in that order

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train: (8131, 91), X_test: (2033, 91)

from sklearn.preprocessing import StandardScaler
numeric_cols = ['year', 'mileage', 'engine']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
```

Think of the four cars as flashcards. Most are kept to study from (train), one is set aside and never looked at while studying (test), then used afterward for an honest quiz. Scaling converts `year`/`mileage`/`engine` from raw units into "how many steps above or below average", so a column with naturally huge numbers (mileage in the tens of thousands) doesn't get treated as more important than a 0/1 column purely because its numbers are bigger.

The split has to happen before scaling: the average and spread used for scaling must come only from the training cars, never influenced by the hidden test cards, that would be a small form of cheating, letting a bit of the answer leak into how the training data gets prepared. The scaler is `fit_transform`'d on train, then only `transform`'d (never re-fit) on test.

Known result to keep in mind: `variant_grouped` alone came out to 72 categories (not the tight grouping originally pictured), which is most of why `X` has 91 columns. Not broken, 8,131 training rows against 91 features is still workable for plain linear regression, but it's a looser fit than ideal, some rare variant columns only have a couple dozen rows behind them, so their coefficients will be noisier. A known limitation, not something fixed today given the timeline.

---

(Next: Stage 6, Modeling. Baseline first (predict the mean, so there is something concrete to beat), then two versions of multiple linear regression side by side, scikit-learn's `LinearRegression`, and the custom `MultipleLinearRegression` class from `learningMultipleLinearRegression.ipynb`, which solves the same problem by hand with the Normal Equation.)

---

## Part 14: Stage 6, Modeling (baseline, scikit-learn, and our own class)

### Baseline model: predicting the mean for every row

```python
baseline_pred = np.full(shape=y_test.shape, fill_value=y_train.mean())
```

Before trusting any real model, first build the laziest possible one: guess the exact same number (the average log price from training) for every single car, no matter its year, mileage, or anything else. This is the bar any real model has to clear. If a fancy model can't beat "just guess the average," it isn't actually learning anything useful.

### scikit-learn's LinearRegression

```python
from sklearn.linear_model import LinearRegression
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
sklearn_pred = sklearn_model.predict(X_test)
```

The trusted, battle-tested library version. Fit on training data only, then used to predict on the test set it has never seen.

### Our own MultipleLinearRegression class (the Normal Equation, by hand)

```python
class MultipleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)
        betas = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coef_) + self.intercept_
        return y_pred

custom_model = MultipleLinearRegression()
custom_model.fit(X_train.astype(float).values, y_train.astype(float).values)
custom_pred = custom_model.predict(X_test.astype(float).values)
```

This does the exact same job as scikit-learn, but by solving the Normal Equation directly: `betas = (X^T X)^-1 X^T y`. In plain words, it inserts a column of 1s at the front (so the math can solve for the intercept alongside every feature's coefficient in one shot), then does matrix multiplication and inversion to solve for every beta (coefficient) at once.

**The error we hit and fixed:** `UFuncTypeError: Cannot cast ufunc 'inv' input from dtype('O') to dtype('float64')`. `X_train` mixes scaled numeric columns (float64, from `year`/`mileage`/`engine`) with one-hot columns that are `True`/`False` booleans. Calling `.values` on a dataframe with mixed float and boolean columns can produce an "object" dtype array (pandas' fallback catch-all type) instead of a clean numeric one, and `np.linalg.inv` refuses to run matrix inversion on anything that isn't purely numeric. Fix: force everything to float first with `.astype(float)` before pulling out `.values`, on both the training call and the predict call.

**Sanity check that confirmed the class works correctly:**

```python
print("sklearn intercept:", sklearn_model.intercept_)
print("custom intercept:", custom_model.intercept_)
# sklearn intercept: 13.988732107757304
# custom intercept:  13.9887321078732
```

Matching to about 7 decimal places is strong evidence the hand-written Normal Equation implementation is mathematically correct, both are solving the exact same underlying optimization problem, just with different code paths (an iterative/optimized library routine vs. direct matrix inversion).

**A red squiggly line that was not a real error:** VS Code's type checker (Pylance) underlined `np.dot(X_test, self.coef_)` in `predict()`, complaining "No overloads for 'dot' match the provided arguments." This is a false alarm, not a runtime bug. Pylance looks at `__init__` and sees `self.coef_ = None`, so it assumes `coef_` could still be `None` by the time `predict()` runs, and it can't prove otherwise just by reading the code. It has no way of knowing that `fit()` always runs first and always replaces `None` with real numbers. The cell actually ran successfully (green checkmark, correct matching intercepts above), so the real Python interpreter never had a problem, only the static analyzer's guess was overly cautious. Lesson: a squiggly underline in the editor is a prediction about what MIGHT go wrong, not proof that something DID go wrong, always check whether the cell actually executed and what it actually printed before assuming a real bug.

(Next: Stage 7, Evaluation. Score baseline, scikit-learn, and our own class against the untouched test set using MAE, MSE, RMSE, R², and Adjusted R², then look at a residual plot to check for any systematic pattern in the errors.)

---

## Part 15: Stage 7, Evaluation & Iteration

### Manual metrics, same pattern as learningSimpleLinearRegression.ipynb

```python
def manual_metrics(y_true, y_pred, name):
    error = y_true - y_pred
    mae = np.abs(error).mean()
    mse = (error ** 2).mean()
    rmse = mse ** 0.5
    print(f"{name}: MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}")
    return mae, mse, rmse

manual_metrics(y_test, baseline_pred, "Baseline (predict the mean)")
manual_metrics(y_test, sklearn_pred, "scikit-learn")
manual_metrics(y_test, custom_pred, "Our own class")
```

Same three-step idea used back in the simple regression notebook: error is just actual minus predicted for every row, MAE is the average size of that error ignoring direction (absolute value), and RMSE squares each error first (which punishes big misses much more than small ones), averages, then square-roots back to the original units. Since `y` here is `log_listingPrice`, these numbers are in log units, not raw rupees, useful for comparing the three models against each other, but not something to read as "off by X rupees" directly.

### Cross-checking with scikit-learn's own metric functions

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

for name, pred in [("Baseline", baseline_pred), ("scikit-learn", sklearn_pred), ("Our own class", custom_pred)]:
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"{name}: MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
```

If the manual numbers above and these library numbers don't match, that is the signal to go back and find a bug in the manual formula, not the other way around, scikit-learn's functions are the trusted reference here.

### R² and Adjusted R²

```python
n = len(y_test)
k = X_test.shape[1]

for name, pred in [("scikit-learn", sklearn_pred), ("Our own class", custom_pred)]:
    r2 = r2_score(y_test, pred)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - 1 - k)
    print(f"{name}: R2={r2:.4f}  Adjusted R2={adjusted_r2:.4f}")
```

R² answers "how much better is this than just guessing the average every time" (the baseline model from Stage 6). A value of 1.0 would be a perfect fit, 0.0 means no better than the baseline, and a negative number would mean the model is actually worse than just guessing the mean.

Adjusted R² exists because R² alone can look better just by adding more columns, even useless, noisy ones, it never goes down when a feature is added, even a garbage one. Adjusted R² applies a penalty based on how many features (`k`) were used relative to how many rows (`n`), which matters a lot here specifically because `X` has 91 columns, many of them from the `variant_grouped` one-hot columns where some categories only have a couple dozen rows behind them.

### Residual plot

```python
residuals = y_test - sklearn_pred
plt.scatter(sklearn_pred, residuals, s=10, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted log(listingPrice)')
plt.ylabel('Residual (Actual - Predicted)')
plt.title('Residual Plot')
plt.show()
```

A residual is just the leftover error for one row, actual minus predicted. Plotting every residual against its predicted value is a pattern check: a healthy model shows a random, formless cloud of dots scattered evenly around the zero line (the red dashed line). If instead there is a visible shape, a curve, a funnel that widens on one side, a slope, that is a sign the model is systematically wrong in some predictable way (for example, consistently underpricing expensive cars), something a single R² number would hide.

(Results and interpretation to be filled in after Restart & Run All, once the actual metric numbers are known.)

### Sample predictions, 30 real cars side by side

Added a comparison table (`Actual Price`, `Predicted Price (sklearn)`, `Predicted Price (our class)`, `Error (sklearn)`) for 30 test-set cars, same idea as `learningSimpleLinearRegression.ipynb`'s comparison dataframe, but with `np.exp()` applied to undo the log transform first, so the table reads in real rupees instead of log units.

### Actual results (Restart & Run All)

```
Baseline (predict the mean): MAE=0.7561  MSE=0.9496  RMSE=0.9745  R2=-0.0001
scikit-learn:                 MAE=0.1742  MSE=0.0735  RMSE=0.2711  R2=0.9226  Adjusted R2=0.9190
Our own class:                MAE=0.1742  MSE=0.0735  RMSE=0.2711  R2=0.9226  Adjusted R2=0.9190
```

R2 of 0.9226 means the model explains about 92% of the variance in log(listingPrice), far above the baseline's -0.0001 (a negative/near-zero R2 confirms the baseline model adds no predictive value, exactly as expected for "always guess the average"). scikit-learn and the custom Normal Equation class produce identical metrics to 4 decimal places, the strongest possible confirmation that the from scratch implementation is correct.

### Closing the loop with Stage 0: RMSE in real rupees

Stage 0 named RMSE in PKR as the metric to report, since a rupee amount is directly interpretable, but every metric above was computed on `log_listingPrice`, the actual training target, so those numbers are in log units. Added a final cell that applies `np.exp()` to the sklearn model's predictions and the true test values, then recomputes MAE and RMSE on the real price scale, directly answering Stage 0's original question ("how many rupees off is this model, typically").

### Multivariate EDA: the skip, made explicit in the notebook itself

Earlier, the Multivariate section header was left with an empty placeholder cell and no explanation, readable in the notebook as unfinished rather than a deliberate choice. Added a markdown cell there explaining the decision (Univariate and Bivariate already surfaced everything Stage 4 and Stage 5 needed, the one-day deadline meant Multivariate combinations were scoped out rather than blocking progress) so anyone reading the notebook standalone, without this NOTES.md file open, understands it was a decision, not a gap.



## Quick Reference: Things I Kept Confusing

A running log of specific questions and confusions from this project, added to every time something trips me up, so I can search this instead of re-asking or re-deriving it from scratch.

**Jupyter only auto-displays the LAST line of a cell.** Put three expressions in one cell and only the last one's result shows up, the first two still ran, their output was just never printed. Fix: `print()` each one, or split into separate cells.

**Mean > median means right-skewed.** A big mountain of common values on the left, a long thin tail of rare extreme values stretching right.

**Histogram bins are equal-width value ranges, not categories.** The x-axis is the value being sliced into ranges, the y-axis is how many rows fall in each range.

**Boxplot anatomy: box = middle 50% (Q1 to Q3), line inside = median, whiskers = normal range (1.5x the box height beyond the box), dots beyond whiskers = outliers.** An outlier is just "statistically unusual within its group," not automatically an error, a car priced far above its group's typical range can be a genuine luxury vehicle, not a data problem. Whether it's an error or genuine depends on context, not the label.

**pandas' `.boxplot()` auto-adds its own title, separate from `plt.title()`.** Without `plt.suptitle('')` to clear it, two titles stack on top of each other.

**`.groupby(col)[other_col].median()` splits rows into buckets by `col`'s values, then computes a stat on `other_col` separately within each bucket.** Nothing calculated until the stat function is called at the end.

**`.where(condition, fallback)` keeps the original value where condition is True, replaces it with fallback where False.** Reads backwards from a normal if-statement's instinct ("do X if true"), it's closer to "protect if true, overwrite if false."

**`pd.get_dummies(..., drop_first=True)` drops one category per column on purpose.** Keeping every category creates a redundant column that's 100% predictable from the others (if a row is 0 in every other category, it must be the missing one), which confuses linear regression's math. The dropped category becomes the implicit baseline, no information is lost.

**Train/test split has to happen before scaling, not after.** Scaling using statistics from the whole dataset lets test-set information leak into how training data gets prepared, giving a dishonestly optimistic sense of how well the model generalizes. Fit the scaler only on training data, apply (never re-fit) to test data.

**Log transform treats equal ratios as equal distances, not equal raw differences.** That's why it compresses large values (like expensive cars) far more than small ones, fixing right-skew, while just dividing by a constant only rescales everything proportionally and keeps the exact same skewed shape.
