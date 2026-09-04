Used Car Price Prediction: Concept Notes

Written to be read cold, by anyone, not just future-me. If you're a beginner reading someone else's GitHub repo, this is meant to walk you through not just what the code does, but how we figured out what to do and why, mistakes included. General ML/stats theory (linear regression, metrics) lives in `../NOTES.md`, this file is specific to real decisions made on this exact dataset.

Dataset: 15,171 real used-car listings scraped from a Karachi marketplace (`used_car_listings_13_12_2025.csv`). Target: `listingPrice`.

---

Part 1: Calling a method vs. just naming it

Before touching duplicates, this tripped us up twice, so it's worth its own section.

```python
def greet():
    return "hello"

greet       # <function greet at 0x...>  , just points at the function, doesn't run it
greet()     # "hello"                     , actually runs it
```

A function or method name on its own is just a label pointing at code sitting in memory. The `()` is the instruction "run this now." This bit us with `df.describe` (no parentheses), it didn't error, it just printed a description of the method object, which happened to include the entire dataframe as part of that description, so it looked like it worked but wasn't actually the statistics table. It bit us again with `df.duplicated.sum()`, except that one raised a real `AttributeError`, because a method object has no `.sum()`, only real data (like a Series) does. Same root bug, two different symptoms. Lesson: if output looks unexpectedly large or generic, check you actually called the method.

---

Part 2: Finding exact duplicates

```python
df.duplicated().sum()
```

`df.duplicated()` returns a `True`/`False` Series, one value per row, `True` if that row is a repeat of an earlier one. Since Python treats `True` as `1` and `False` as `0`, summing that Series just counts how many rows got flagged.

Result: 2,477 exact duplicate rows, about 16% of the entire dataset. Worth sitting with that number before doing anything about it; that's not noise, that's one row in six.

The `keep` parameter

`duplicated()` always spares exactly one occurrence per repeated group and flags the rest as `True`, `keep` controls which occurrence is spared:

| values in order | `keep='first'` (default) | `keep='last'` | `keep=False` |
|---|---|---|---|
| `A, A, B, A, C` (rows 0-4) | row 0 spared; rows 1, 3 → `True` | row 3 spared; rows 0, 1 → `True` | nobody spared; rows 0, 1, 3 → `True` |

For counting duplicates, the default (`keep='first'`) is fine, you just want "how many extra copies exist." For looking at a duplicate group with your own eyes, you need `keep=False`, otherwise the very first (and "spared") copy gets filtered out of your view, and you're left staring at repeats with nothing to compare them against.

Looking at the actual rows

```python
df[df.duplicated(keep=False)].sort_values(by=['manufacturer','variant','year','mileage']).head(20)
```

Read this inside-out: `df.duplicated(keep=False)` makes the `True`/`False` map → `df[ ... ]` filters `df` down to only the flagged rows → `.sort_values(by=[...])` reorders them so matching rows land next to each other instead of being scattered by original row position → `.head(20)` caps the output to a manageable sample.

This is a reusable pattern, not just a duplicates thing:
```python
df[ <condition that picks the rows you care about> ].sort_values(by=[ <columns that cluster related rows> ]).head(N)
```

What we found, and how we figured out what it meant

Every duplicate pair we looked at had identical mileage, down to the exact kilometer (e.g. two separate rows both reading `45,812 km`), and the two copies of each pair sat far apart in the original row order (sometimes thousands of rows apart).

Two clues, one conclusion: mileage is a continuous, highly specific number, two genuinely different cars, sold by two different people, coincidentally sharing the exact same odometer reading is essentially implausible. And if the duplication came from something like the scraper hitting the same page twice in a row, you'd expect the copies to sit close together in row order, not scattered across the whole file. Far-apart row positions plus an exact match on a high-cardinality number like mileage points at one thing: the same listing got captured more than once during separate scraping passes, a data-collection artifact, not two real cars that happen to be identical.

---

Part 3: Finding near-duplicates

Exact-duplicate detection only catches rows where every single column matches. But what about the same physical car, listed once, then listed again months later after being driven a bit more, mileage and maybe price would differ, so `duplicated()` would never flag it. That needs a looser check.

```python
df[df.duplicated(subset=['manufacturer','variant','year','engine','fuelType','transmission'], keep=False)] \
    .sort_values(by=['manufacturer','variant','year','engine','fuelType','transmission']).head(20)
```

`subset=[...]` tells `duplicated()` to only compare the listed columns, everything else is ignored. Deliberately left out of the subset: `mileage` and `listingPrice`, those are exactly the two things we expect might legitimately differ between two snapshots of the same real car, and if we required them to match too, we'd miss the pattern we're trying to catch.

Two very different things came back, and they need different handling

Audi A3 2016 (3 rows, all `Petrol`/`Automatic`/`1200cc`): mileages `101,000` / `41,000` / `62,000 km`, prices `5,950,000` / `6,700,000` / `5,850,000`, no two agree on anything. Conclusion: this is not duplication. Three different sellers, three different actual cars, that happen to share a common configuration. Real, legitimate, distinct data, must not be dropped.

Audi A4 2014 (part of the group): an already-known exact-duplicate pair (`48,000 km`, `6,500,000`, twice) plus one more row at `48,810 km` and the same `6,500,000` price. Off by 810 km, identical price. That pattern, a small, plausible, real-world mileage increase with an unchanged asking price, is a much better match for "the same car, seen again months later" than for a scraper bug. (A bug would more likely reproduce numbers exactly, the way the true exact-duplicate pair already does, not invent a believable 810 km bump on its own.)

Takeaway: "near-duplicate" isn't one category. It splits into at least two: coincidental matches on a popular configuration (keep, they're real), and genuine same-car-at-different-times matches (probably redundant, worth merging).

The cleaning decision, and why "good enough, documented" was the right call here

We could try to build a fully general "is this really the same car" classifier. For a first project, that's disproportionate, this dataset has no seller ID or listing ID to confirm identity with certainty, so perfect confidence isn't available at any amount of effort, only diminishing returns.

The rule we're using instead: a near-duplicate group counts as "same car" only if price matches exactly and mileage is within a small gap (2,000 km, comfortably above the genuine case we found at 810 km, comfortably below the coincidence case's tens-of-thousands-km gaps). Anything that doesn't clear that bar is left alone as legitimate distinct data.

This is a deliberate, stated simplification, not a hidden shortcut, and that distinction matters for a portfolio project. "I used a documented mileage-gap heuristic, and here's its known limitation" reads as engineering judgment. Silently pretending the cleaning is perfect, or burning days chasing certainty the data can't provide, both read worse.

---

Part 4: Converting text columns into real numbers

Before this step: `df_clean = df.drop_duplicates(keep='first')` dropped the 2,477 exact duplicates (verified: `df_clean.shape` → `(12694, 10)`). But `mileage` (`"156,000 km"`) and `engine` (`"1800cc"`) are still text, not numbers, and no math (comparisons, model training, `groupby` aggregation) works correctly on text that merely looks like a number. Rule of habit going forward: before transforming or comparing any column, check `.dtype`, if it says `object`/`str` but the values look numeric, that's a sign it needs converting first.

Mileage

```python
df_clean['mileage'] = df_clean['mileage'].str.replace(',', '').str.replace(' km', '').astype(int)
```

`.str` unlocks string operations across an entire column at once, instead of writing a loop. `.str.replace(',', '')` deletes every comma; `.str.replace(' km', '')` deletes the unit suffix; `.astype(int)` converts the now-clean text (`"156000"`) into an actual integer (`156000`).

Engine: the same idea, plus a real landmine

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

The actual fix:

```python
df_clean['engine'] = df_clean['engine'].str.replace('cc', '')
df_clean.loc[df_clean['engine'] == '', 'engine'] = '0'
df_clean['engine'] = df_clean['engine'].astype(int)
```

Traced through one row of each kind:

| | after line 1 (`str.replace('cc','')`) | after line 2 (`.loc[...]='0'`) | after line 3 (`.astype(int)`) |
|---|---|---|---|
| Toyota, was `"1800cc"` | `"1800"` | unchanged, condition is `False` for this row | `1800` |
| Nissan Leaf, was `"cc"` | `""` (empty) | condition is `True` → becomes `"0"` | `0` |

`.loc[<condition>, '<column>'] = <value>` is the general recipe for "find exactly these rows, and only in this one column, write this value", different from `df[condition]`, which only lets you look, not change anything. This is a pattern worth reusing any time a small subset of rows needs fixing rather than the whole column.

A subtlety caught along the way: an earlier attempt used `.str.replace('cc', ' ')` (a space) instead of `''` (nothing). It happened to still work, because Python's `int()` quietly strips whitespace, but that's the code working by accident, not by design. Worth remembering: "it ran without an error" isn't the same as "it did what I meant."

---

Part 5: Finishing Stage 2: near-duplicates, and dropping `age`

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

(Next: Stage 3, EDA. Univariate first: look at each column on its own, starting with `listingPrice`'s distribution, before comparing columns against each other.)

