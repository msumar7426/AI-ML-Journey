# Gradient Descent: Concept Notes

Status: blocked, not started.

`learningGradientDescent.ipynb` in this folder is currently a 0-byte empty file, there's no code here yet to ground these notes in, and writing "explain your own code" notes without your actual code would defeat the point of this file. This was flagged earlier as an open item and hasn't been resolved.

Once the notebook has real content again (either recovered from wherever it went, or redone), this file should cover, in the same style as `../../notes/NOTES.md`:
- Why gradient descent exists as an alternative to the Normal Equation (the Normal Equation needs to invert `XᵀX`, which gets slow/unstable as the number of features grows, gradient descent scales better and doesn't require that inversion at all).
- The update rule, in plain words: what the learning rate controls, why too large a value overshoots and too small one crawls.
- The from-scratch implementation, annotated like the Linear Regression classes.
