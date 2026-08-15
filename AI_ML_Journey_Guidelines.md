# AI/ML Journey Guidelines

> **Purpose:** These guidelines define how the AI/ML Journey will be learned, implemented, documented, reviewed, and developed into a job-ready portfolio.

---

## 1. Mission

The goal of this repository is not simply to complete the CampusX 100 Days of Machine Learning playlist.

The goal is to become capable of:

- Understanding core Machine Learning concepts deeply.
- Writing ML code independently.
- Implementing important algorithms from scratch.
- Using industry-standard libraries correctly.
- Evaluating models using appropriate metrics.
- Improving model performance systematically.
- Choosing appropriate algorithms for real-world problems.
- Explaining ML concepts in interviews.
- Building and documenting practical projects.
- Maintaining a professional GitHub portfolio.

**Mastery is more important than speed.**

---

# 2. Core Learning Philosophy

### Rule 1 — One topic at a time

We focus on one major ML topic until its important concepts are understood.

We do not rush through the playlist simply to maintain the 100-day schedule.

### Rule 2 — Understand before memorizing

We prioritize:

> Intuition → Understanding → Mathematics → Code → Practice → Recall

over:

> Memorization → Copying code → Moving on

### Rule 3 — CampusX provides the learning path

CampusX is the primary course/resource guiding the sequence of topics.

The mentor workflow expands each topic beyond the video where necessary.

### Rule 4 — Learn what is relevant

We do not study entire subjects unnecessarily.

If Linear Regression requires derivatives, we learn the relevant derivative concepts.

We do not stop ML to complete an entire calculus course.

---

# 3. Standard Topic Workflow

Every major CampusX topic follows this process:

```text
1. Watch the CampusX video
        ↓
2. Code along in VS Code
        ↓
3. Make first-pass notes
        ↓
4. Explain what was learned
        ↓
5. Identify questions/confusion
        ↓
6. Python prerequisite analysis
        ↓
7. Mathematics prerequisite analysis
        ↓
8. Fill conceptual gaps
        ↓
9. Deep ML study
        ↓
10. From-scratch implementation
        ↓
11. Library implementation
        ↓
12. Visualization / experiments
        ↓
13. Evaluation
        ↓
14. Performance improvement
        ↓
15. Model-selection reasoning
        ↓
16. Industry applications
        ↓
17. Interview preparation
        ↓
18. Mini exercise/project
        ↓
19. Code review
        ↓
20. Documentation
        ↓
21. Git commit + push
```

A topic is considered complete only when the important parts of this workflow have been addressed.

---

# 4. Python Prerequisite Analysis

Before implementing an ML topic, identify the Python concepts required.

## 🟢 Already Known

Concepts the learner already understands sufficiently.

Do not waste time relearning them unless they become a problem.

Examples:

- Variables
- Conditions
- Loops
- Functions
- Basic data structures

## 🟡 Strengthen

Concepts that are known but need additional practice.

Examples:

- NumPy array operations
- Functions with multiple arguments
- List comprehensions
- Matrix operations

## 🔴 Must Learn

A Python concept that directly blocks understanding of the current ML implementation.

Examples:

- Classes
- `__init__`
- `self`
- Object attributes
- Instance methods

If a concept is blocking progress, pause and learn it before continuing.

## 📌 Learn Later

Useful concepts that are not necessary for the current topic.

Do not allow them to derail the current learning objective.

### Anti-Rabbit-Hole Rule

When an unfamiliar Python feature appears:

> Learn only what is required to understand the current ML topic unless the feature itself is a learning objective.

---

# 5. Mathematics Prerequisite Analysis

Mathematics will be taught using a **just-in-time approach**.

We learn mathematics when it becomes useful for understanding an ML concept.

## Required explanation order

For mathematical concepts:

```text
Real-world intuition
        ↓
Simple example
        ↓
Visual interpretation
        ↓
Basic mathematical notation
        ↓
Formula
        ↓
Connection to ML
        ↓
Implementation
```

Never begin with a complicated formula when a simple intuition can explain the idea first.

## Mathematics categories

### 🟢 Essential Now

Mathematics required to understand the current topic.

### 🟡 Useful Soon

Mathematics likely to become important for upcoming ML topics.

### 📌 Learn Later

Mathematics that is interesting but not currently necessary.

### Examples

For Linear Regression:

- Basic algebra
- Functions
- Slope
- Mean
- Squared error
- Derivative intuition
- Partial derivative intuition
- Chain rule intuition
- Gradient Descent

Not required at this stage:

- Advanced integration
- Differential equations
- Proof-heavy calculus
- Advanced mathematical theory

---

# 6. ML Algorithm Mastery Framework

Every important algorithm should eventually be understood through the following questions.

## 6.1 Problem

- What problem does it solve?
- Regression, classification, clustering, dimensionality reduction, etc.?
- What type of target does it require?

## 6.2 Intuition

- What is the algorithm trying to accomplish?
- Why does it work conceptually?
- What is the simplest mental model?

## 6.3 Mathematics

- What mathematical ideas power it?
- What formulas actually matter?
- Why are those formulas necessary?

## 6.4 Assumptions

- What assumptions does the model make?
- What happens when those assumptions are violated?

## 6.5 From Scratch

Implement the important mechanism ourselves where practical.

The goal is understanding, not reinventing every production library.

## 6.6 Libraries

Use appropriate industry-standard tools such as:

- NumPy
- Pandas
- Matplotlib
- scikit-learn

Later, additional libraries will be introduced when required.

## 6.7 Visualization

Use visualization when it improves understanding.

Examples:

- Regression lines
- Residuals
- Decision boundaries
- Trees
- Feature importance
- Learning curves
- Confusion matrices

## 6.8 Evaluation

Understand:

- Which metrics are available?
- What does each metric measure?
- When is each metric appropriate?
- What can make a metric misleading?

## 6.9 Improvement

Investigate model performance systematically rather than randomly changing parameters.

Possible improvement areas:

- Data quality
- Missing values
- Outliers
- Feature engineering
- Feature selection
- Encoding
- Scaling
- Model choice
- Hyperparameters
- Cross-validation
- Error analysis

## 6.10 Failure Modes

For every important algorithm:

> When does this model perform badly, and why?

## 6.11 Model Selection

Never memorize:

> "Algorithm X is best."

Instead ask:

- What is the problem?
- What is the target?
- How much data is available?
- What type of features exist?
- Are relationships linear or nonlinear?
- Are classes balanced?
- Is interpretability important?
- Are training time and inference speed important?
- Are there missing values?
- What baseline should we establish?

---

# 7. Model Evaluation Framework

Evaluation must match the problem.

## Regression

Important metrics include:

- MAE
- MSE
- RMSE
- R²

We will learn what each metric means and when it is useful.

## Classification

Important tools include:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Log Loss

### Balanced Classes

Accuracy can be useful when class distribution is reasonably balanced, but it should not automatically be the only metric.

### Imbalanced Classes

Accuracy can be highly misleading.

We will focus more on:

- Precision
- Recall
- F1
- PR-AUC
- Confusion Matrix

depending on the business/problem context.

### Business-cost rule

Metric selection should reflect the consequences of errors.

Ask:

> Is a false positive worse, or is a false negative worse?

This question often determines which metric matters most.

---

# 8. Model Improvement Framework

A model should not be improved by blindly changing parameters.

Use this sequence:

```text
Establish baseline
      ↓
Evaluate correctly
      ↓
Inspect errors
      ↓
Check data quality
      ↓
Improve features/preprocessing
      ↓
Try appropriate models
      ↓
Tune hyperparameters
      ↓
Cross-validation
      ↓
Compare fairly
      ↓
Final evaluation
```

## Hyperparameter tuning

We will learn:

- What hyperparameters are.
- Which hyperparameters matter for each algorithm.
- Why changing them affects performance.
- Grid Search.
- Randomized Search.
- Cross-validation.

## Critical rule

A higher score on one split does not automatically mean a better model.

We must consider:

- Validation strategy
- Generalization
- Overfitting
- Data leakage
- Appropriate metrics

---

# 9. Model Selection Framework

When presented with a problem, do not immediately choose an algorithm.

First perform **problem framing**.

Example:

> Predict the 2027 Formula 1 champion.

Ask:

1. What exactly are we predicting?
2. Before or during the season?
3. Race winner or championship winner?
4. Classification or regression?
5. What historical features are available?
6. How much training data exists?
7. Are predictions made once or repeatedly?
8. What metric should evaluate success?
9. What baseline should we compare against?
10. What models are reasonable candidates?

Only then choose and compare models.

The objective is to develop **model-selection reasoning**, not algorithm memorization.

---

# 10. Industry Perspective

For important topics, answer:

- Where is this used in industry?
- Why would a company choose it?
- What alternatives exist?
- What are its operational limitations?
- What happens after training?
- How would the model be deployed?
- How would performance be monitored?

We will gradually introduce:

```text
Data
 ↓
Preprocessing
 ↓
Training
 ↓
Validation
 ↓
Tuning
 ↓
Evaluation
 ↓
Deployment
 ↓
Monitoring
```

Not every topic requires the entire production pipeline immediately.

---

# 11. Notebook Strategy

The physical notebook is for **high-value knowledge that should remain memorable**.

We will explicitly mark:

### ✍️ WRITE THIS

Important information worth writing by hand.

### 💡 INTUITION

The mental model that makes the concept easier to remember.

### 🧠 REMEMBER

A short rule or relationship worth retaining.

### ⚠️ COMMON MISTAKE

A misconception or error to avoid.

### 🎤 INTERVIEW TIP

A useful interview insight.

### 🔧 INDUSTRY TIP

A practical engineering consideration.

Do not fill the notebook with every line of code or every derivation.

The notebook should become a **personal ML handbook**, not a transcript of the course.

---

# 12. Coding Standards

Development environment:

- VS Code
- Python virtual environment
- Git
- GitHub

Code should prioritize:

- Readability
- Meaningful names
- Small, understandable functions
- Appropriate comments
- Reproducibility
- PEP 8 conventions

Formatting/tools:

- Black
- isort
- Pylance
- Error Lens
- GitLens

Code should be written by the learner rather than blindly copied.

---

# 13. From-Scratch vs Library Implementation

Both are important, but they serve different purposes.

## From Scratch

Purpose:

> Understand how the algorithm works.

Example:

```python
class LinearRegression:
    ...
```

using NumPy where appropriate.

## Library

Purpose:

> Learn how the algorithm is actually used in practical ML workflows.

Example:

```python
from sklearn.linear_model import LinearRegression
```

We compare:

- Our implementation
- Library implementation
- Results
- Limitations
- API design

We do not expect to reproduce the engineering complexity of mature libraries.

---

# 14. Interview Preparation

Each major topic should eventually include:

### Beginner questions

Definitions and basic intuition.

### Intermediate questions

Mechanisms, trade-offs, and mathematics.

### Practical questions

Debugging, evaluation, preprocessing, and model improvement.

### Advanced questions

Failure modes, assumptions, optimization, and model selection.

### Scenario questions

Example:

> Your model has 95% accuracy but performs poorly in production. What would you investigate?

The goal is to develop reasoning rather than memorize answers.

---

# 15. Projects

Projects will progress in complexity.

```text
Exercises
   ↓
Mini Exercises
   ↓
Mini Projects
   ↓
End-to-End Projects
   ↓
Portfolio Projects
```

Projects should demonstrate:

- Problem definition
- Data understanding
- Exploratory analysis
- Preprocessing
- Baseline
- Model selection
- Training
- Evaluation
- Improvement
- Error analysis
- Documentation

---

# 16. Git & GitHub Workflow

Each meaningful learning milestone should be documented.

Preferred commit style:

```text
feat: implement linear regression from scratch
docs: document linear regression concepts
fix: correct gradient descent update
refactor: simplify regression implementation
chore: update project structure
```

Commits should be:

- Small
- Meaningful
- Descriptive
- Related to one logical change

Avoid commits such as:

```text
update
stuff
final
changes
asdf
```

---

# 17. Definition of "Mastered"

A topic is not mastered merely because:

- The CampusX video was watched.
- The code runs.
- The model produces a score.

A topic is considered sufficiently mastered when the learner can:

### Explain

Explain the concept in simple language.

### Understand

Describe the intuition and important mathematics.

### Implement

Write an important version from scratch.

### Use

Use the appropriate library correctly.

### Evaluate

Choose appropriate metrics and interpret results.

### Improve

Identify reasonable ways to improve performance.

### Select

Explain when to use the algorithm and when not to use it.

### Diagnose

Identify common reasons for poor performance.

### Interview

Answer conceptual, mathematical, coding, and scenario-based questions.

### Apply

Use the algorithm on a meaningful dataset or project.

---

# 18. The "Don't Get Lost" Rule

When something unfamiliar appears:

### Ask:

> Is this necessary to understand the current ML concept?

If **yes**:

> Learn it now.

If **no**:

> Understand enough to avoid confusion and mark it for later.

This prevents endless detours such as:

```text
Linear Regression
    ↓
Classes
    ↓
OOP
    ↓
Inheritance
    ↓
Decorators
    ↓
Metaclasses
    ↓
...
```

We are building ML expertise, not trying to learn the entire Python language before completing one algorithm.

---

# 19. Depth Over Speed

The 100-day schedule is a **roadmap, not a deadline**.

If one important concept takes several study sessions to understand properly, that is acceptable.

The objective is:

> **Become capable, not merely complete.**

---

# 20. Final Principle

Every major concept should answer five questions:

> **What is it?**

> **How does it work?**

> **How do I implement it?**

> **When should I use it?**

> **How do I know whether it is working well?**

If we can answer all five, we are learning Machine Learning rather than simply watching Machine Learning tutorials.

---

## Our commitment

This repository will document the progression from beginner-level foundations to practical AI/ML engineering.

The priority order is:

**Understanding → Implementation → Evaluation → Improvement → Application → Communication**

Not:

**Speed → Completion → Certificates**
