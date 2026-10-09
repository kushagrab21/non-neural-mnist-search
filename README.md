# Non-neural MNIST classifier found by LLM-agent search

This repository holds a self-contained digit classifier, 27,654 bytes of Python and NumPy, with no neural network and no external files. It was produced by a three-role LLM-agent search loop for the DSA5105 after-class experiment (lecture slides 65–67).

The full write-up is in [REPORT.pdf](REPORT.pdf). It covers the problem, the generating workflow, the three iterations, the test comparison with a small CNN, and a dissection of the classifier.

## Results

| On the 10,000 MNIST test images | Searched program | Small CNN |
|---|---:|---:|
| Test accuracy | **98.09%** | **98.83%** |
| Errors | 191 | 117 |
| Size of the distributed model | 27,654 B | 265,390 B |
| Latency per image (development measurement) | 0.30 ms | 0.50 ms |
| Runtime dependencies | NumPy | NumPy, PyTorch |

Each program was frozen before the test set was opened, and each was run on it once. The latency was measured earlier on development images; it includes inter-process overhead. See Section 5 of the report.

## Using the classifier

```python
import numpy as np
from classifier.classifier import predict

image = ...                 # uint8 array of shape (28, 28), an MNIST-style digit
digit = predict(image)      # int from 0 to 9
```

It needs Python 3.11 and NumPy (tested with 2.3.5). Every learned value is embedded in `classifier/classifier.py`, so nothing else is loaded.

## Contents

| Folder | What it holds |
|---|---|
| `classifier/` | The submitted classifier, which is iteration 2 |
| `iterations/` | The three programs found by the search. Each folder has the emitted `classifier.py`, the `build.py` that the agent wrote and that fitted it, and the build's own report |
| `baseline_cnn/` | The comparison CNN: its architecture, prediction wrapper, weights and training script |
| `workflow/` | The orchestrator code that ran the Researcher, Runner and Controller sessions, the build sandbox and the evaluator |
| `analysis/` | The follow-up analysis behind Section 7 of the report: its code and its comparison tables |
| `results/` | Per-digit test accuracy of the submitted classifier |

## Not included

- The MNIST data.
- The raw run records and agent transcripts.

The workflow and analysis code are published as they were run. They depend on the original local setup (Docker images, the Claude Code CLI and a prepared data split), so they document the method rather than reproduce it in one command. The classifier and the CNN run on their own.
