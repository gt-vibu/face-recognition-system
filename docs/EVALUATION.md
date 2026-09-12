# Evaluation

**This document must be filled in with results actually produced by running
`python evaluation/run_evaluation.py` against your own collected data. Do not
fabricate numbers — an honest small-scale evaluation is worth far more, and
is far more defensible in an interview, than invented "clean" results.**

## Dataset

- Known identities: _(e.g. 5 people)_
- Enrollment images per known person: _(e.g. 3, named `enroll_1.jpg`, `enroll_2.jpg`, ...)_
- Held-out test images per known person: _(e.g. 2-3, named `test_1.jpg`, ...)_
- Unknown identities (never enrolled): _(e.g. 2-3 people)_, images under `evaluation_data/unknown/`

Enrollment and test images for the same person must be **different photos** — never evaluate on the exact images used to build the reference embedding.

## Method

1. Build one averaged reference embedding per known person from their `enroll_*` images.
2. For every held-out `test_*` image of a known person, compute similarity against every reference:
   - Same person → **genuine** score
   - Different person → **impostor** score
3. For every unknown-identity image, compute similarity against every reference → **unknown** score.
4. Report the distribution of each score group, plus FAR/FRR at the current `CONFIRMED_THRESHOLD`.

## Results

_(Paste the actual console output of `evaluation/run_evaluation.py` here.)_

```
Genuine:  n=__  min=__  max=__  mean=__
Impostor: n=__  min=__  max=__  mean=__
Unknown:  n=__  min=__  max=__  mean=__

At CONFIRMED_THRESHOLD = __:
False Acceptance Rate (FAR): __
False Rejection Rate (FRR): __
```

## Threshold selection reasoning

_(Explain, using the numbers above: where did genuine and impostor/unknown scores cluster? What gap did you find? Why did you pick the final `CONFIRMED_THRESHOLD` and `UNCERTAIN_LOWER_BOUND` values in `src/config.py`? Update the config file to match, and log the decision in `docs/DECISIONS.md`.)_

## Observations & limitations

_(Anything notable: which people were hardest to distinguish, which photos performed worst, whether the dataset size felt sufficient, etc.)_
