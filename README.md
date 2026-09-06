# AI and Machine Learning Journey

> A structured, hands on roadmap documenting my journey to becoming an AI/ML Engineer.

## About

Hi, I am M S Umar, a Computer Science student passionate about Artificial Intelligence, Machine Learning, and building impactful software.

This repository documents my learning journey through theory, implementation, real world projects, and interview preparation. Each topic is learned from CampusX's 100 Days of Machine Learning course, then extended with from scratch implementations and a real dataset project before moving to the next topic.

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

## Repository Structure

```text
AI-ML-Journey/
  docs/AI_ML_Journey_Guidelines.md  how each topic is learned, step by step
  templates/                       reusable templates (e.g. the ML project workflow)
  01_Linear_Regression/            Simple and Multiple Linear Regression, Gradient Descent
    notes/                         concept notes and reference material for this topic
    03_Gradient_Descent/           gradient descent (in progress)
      notes/
    project/                       real project: used car price prediction
      notes/
    deployment/                    Streamlit frontend for the project above, sibling to project/
  02_Uni&MultivariateAnalysis/     univariate and multivariate EDA, pandas profiling
```

Every code folder keeps a `notes/` subfolder alongside it. That is where the concept explanations, worked examples, and reasoning behind every decision live, written to be read cold by a beginner. This README is the only file in the repository meant to pitch the project; everything under `notes/` is study material for me (or for you, if you found this repo and want to learn alongside it).

## Projects

| Project | Status | Description |
|---|---|---|
| Used Car Price Prediction | Complete (8 of 8 stages), with a deployable frontend | Multiple linear regression on 15,171 real used car listings scraped from a Karachi marketplace. R2 of 0.92 on held out test data. Includes a Streamlit frontend (`01_Linear_Regression/deployment/`), free to run locally or deploy. See `01_Linear_Regression/project/README.md` for the full writeup, and `01_Linear_Regression/project/notes/` for the reasoning behind every decision. |

## Progress Tracker

| Module | Status |
|---|---|
| Environment Setup | Done |
| Git and GitHub | Done |
| Simple and Multiple Linear Regression (theory and from scratch code) | Done |
| Gradient Descent | In progress |
| Used Car Price Prediction project | Done |
| Univariate and Multivariate Analysis | In progress |

## Learning Approach

Every topic follows this workflow:

1. Learn the theory.
2. Understand the mathematics.
3. Implement in Python.
4. Build from scratch.
5. Apply to a real dataset.
6. Solve a mini exercise.
7. Document key learnings.
8. Commit and push to GitHub.

The full version of this workflow, including how to triage Python and math prerequisites for a new topic, is in `docs/AI_ML_Journey_Guidelines.md`.

## Where This Is Going

Each topic gets the same treatment as linear regression did: theory, a from-scratch implementation, and a real dataset project. Deep learning, NLP and deployment come after the fundamentals are solid.

## Contact

- GitHub: https://github.com/msumar7426
- Email: muhammadsiddiqueumar1@gmail.com
