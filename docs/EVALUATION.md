# Evaluation

All numbers in this document were produced by actually running the system — the Next.js UI and FastAPI
backend on a scratch database, `evaluation/run_evaluation.py`, and the same `src/` functions the app uses.
Nothing here is estimated or invented.

## Read this first — honest caveats

- **Real-person evidence comes from two sources:** a 50-person sample of the public LFW dataset (250 real photos,
  section 0) and two selfies of one person ("RealB") used in the in-app validation (sections 1–9).
- **The rest of the in-app validation uses synthetic (AI-generated) identities.** Generated "different people" are
  not guaranteed to be different identities — the generator reused faces (see "Synthetic data: duplicate faces"
  below), which produces false accepts that do not occur between real, unrelated people.
- **Robustness on RealB was measured on programmatic variants** (darker, blurred, rotated, occluded…) of the two
  enrollment selfies. They are *not* independent photos, so they overstate real-world robustness.
- **Conclusion:** on 50 real people the system made no wrong-person confirmations and confirmed no strangers at the
  default thresholds; borderline genuine photos go to Uncertain rather than being mis-identified. LFW photos are
  mostly frontal press photos of random (not look-alike) people, so this is **not** proof of real-world accuracy on
  phone/webcam photos, look-alikes, twins or relatives.

## 0. Real-person evaluation — LFW, 50 people

**Dataset.** Labeled Faces in the Wild (deep-funneled version, Kaggle `jessicali9530/lfw-dataset`). A random sample
(fixed seed) of **50 people who have at least 5 photos, using their first 5 photos — 250 real photos.** The images
are not stored in this repository.

**Method — the app's own pipeline and rules.**
- Same model and settings (`buffalo_l`, `DET_SIZE` 640, CPU). The embeddings were checked against
  `src.embeddings.detect_faces` (cosine similarity ≈ 1.0).
- Same enrollment: 2 photos averaged and re-normalised; a photo with 0 or more than 1 detected face is rejected,
  as in the app. Every 2-photo combination per person (10) was used, with the other 3 photos as probes.
- Same decision: best match over the gallery, Confirmed ≥ 0.62, Uncertain 0.45–0.62, Unknown < 0.45.
- **Unknown people:** leave-one-person-out — each person's photos are identified against a gallery *without* them.
- A face was detected in 248 / 250 photos. 35 photos contain a second (background) face, so 131 of the 500
  person-enrollments were skipped, as the app would reject them.

**Samples.** 1,101 genuine identification decisions, 39,560 impostor comparisons (probe vs another enrolled
person), 1,839 unknown-person decisions; plus 493 genuine and 30,135 impostor single-photo pairs.

**Score distributions (2-photo references)**

| | Samples | Similarity range |
|---|---|---|
| Genuine (probe vs own reference) | 1,101 | **0.456 – 0.895** (mean 0.728) |
| Impostor (probe vs another enrolled person) | 39,560 | −0.195 – **0.313** (mean 0.007) |
| Unknown person, best match in gallery | 1,839 | 0.038 – 0.313 |
| Overlap between genuine and impostor | — | **none** (gap 0.313 → 0.456) |

Single photo vs single photo: genuine 0.383 – 0.913, impostor −0.212 – 0.296, again no overlap.
The top-ranked person was correct in **100%** of the 1,101 genuine decisions.

**Decisions by Confirmed threshold** (Uncertain lower bound fixed at 0.45)

| Confirmed threshold | Correctly Confirmed | Confirmed as the wrong person | Uncertain (retry) | Genuine → Unknown | Unknown person Confirmed |
|---|---|---|---|---|---|
| 0.45 | 100% | 0 | 0% | 0% | 0 |
| 0.55 | 97.7% | 0 | 2.3% | 0% | 0 |
| 0.60 | 93.3% | 0 | 6.7% | 0% | 0 |
| **0.62 (default)** | **90.2%** | **0** | **9.8%** | **0%** | **0** |
| 0.65 | 83.7% | 0 | 16.3% | 0% | 0 |
| 0.70 | 64.9% | 0 | 35.1% | 0% | 0 |

At 0.62 on single-photo pairs: TA 325, FR 168, TR 30,135, FA 0 → FAR 0%, FRR 34.1% (a single photo is a much weaker
reference than the 2-photo average the app stores).

**Number of enrollment photos** (same people, the remaining photos as probes, default thresholds)

| Enrollment photos | Genuine decisions | Correctly Confirmed at 0.62 | Uncertain | Wrong person / Unknown person Confirmed |
|---|---|---|---|---|
| 2 | 1,101 | 90.2% | 9.8% | 0 / 0 |
| **3** | 652 | **95.9%** | 4.1% | 0 / 0 |
| 4 | 149 | 97.3% | 2.7% | 0 / 0 |

→ The Enroll page now recommends **3 photos with slightly different angles or lighting** (2 remains the minimum;
the app already averages any number of accepted photos from 2 to 5).

**Tried and not adopted.** Horizontal-flip test-time augmentation (averaging each face with its mirror image) gave
92.0% Confirmed at 0.62 but pushed 2 of 1,101 genuine photos below 0.45 (Unknown), and would change the embedding
code — not worth it.

**Limits of this test.** 50 random people are not look-alikes; twins/relatives were not tested; LFW photos are
mostly frontal, well-lit press photos (easier than phone/webcam photos); the 10 enrollment rotations reuse the same
photos, so decisions are not independent; 0 false accepts in 39,560 comparisons does not prove the rate is zero.

## Synthetic data: duplicate faces

Similarity between the averaged references of synthetic identities that are labelled as *different* people:

| Pair | Similarity |
|---|---|
| SynthA ~ "B" | 0.756 |
| P1 ~ P5 / P1 ~ P3 / P3 ~ P5 | 0.760 / 0.696 / 0.648 |
| P2 ~ P4 | 0.660 |
| P1, P3, P5 ~ SynthA and "B" | 0.51 – 0.55 |

For comparison, real unrelated people never exceed **0.313** (LFW, above), and RealB vs every synthetic face stays
≤ 0.126. The "5-face" synthetic composite is SynthA four times (0.86–0.90) plus "B" (0.976). No image file is a
copy of another (checked by hash and perceptual hash). Every false accept in sections 3–4 comes from these
duplicate-face identities.

## In-app validation — dataset

| Label | Source | Count | Used for |
|---|---|---|---|
| `real_*` | Real photos of one person (RealB) | 2 | enrollment; real "unknown" before enrollment |
| `aug_*` | Programmatic variants of the 2 real photos | 26 (13 per photo) | lighting / blur / rotation / occlusion robustness |
| SynthA | AI-generated woman (4 photos) | 4 | enrollment (2), different-photo tests (2) |
| "B" | AI-generated "different person" that reuses A's face | 1 | look-alike impostor |
| P1–P5 | AI-generated people, 3 photos each | 15 | P1, P2 enrolled; P3, P4, P5 never enrolled |
| Gemini woman | AI-generated, frontal / three-quarter / profile | 3 | unknown person at different angles |
| Composites | 2-face and 5-face images; background-only crop; blank; `.txt` | 5 | failure cases |

Enrollment for the identification tests used **2 photos per person**, uploaded through the Next.js Enroll page
(SynthA, P1, P2, RealB). The stored references were checked directly in SQLite: 512-dimensional embeddings with
an L2 norm of exactly `1.000000`, `num_samples = 2`.

## In-app validation — method

1. Enroll through the Next.js **Enroll** page (FastAPI → `detect_faces` → average + re-normalise → SQLite).
2. Identify through the Next.js **Identify** page and the same `/api/identify` endpoint (via the Next.js proxy)
   for exact 4-decimal scores. The decision is always made by `src/matching.best_match`.
3. For the impostor matrix, compute the cosine similarity of every photo against **every** stored reference
   with the app's own `detect_faces` + `cosine_similarity` (read-only on the scratch database).
4. Run the existing `evaluation/run_evaluation.py` on `evaluation_data/` (P1–P5 with 2 enrollment + 1 test photo
   each, the two real selfies as "unknown").
5. Sweep the Confirmed threshold over 0.50–0.70 on the recorded scores (the Uncertain lower bound fixed at 0.45),
   and verify with the evaluation slider that the running app produces the same decisions.

## In-app validation — results

### 1. `evaluation/run_evaluation.py` (actual console output)

```
Built reference embeddings for: ['person1', 'person2', 'person3', 'person4', 'person5']

--- Raw score distributions ---
Genuine: n=5  min=0.580  max=0.637  mean=0.610
Impostor: n=20  min=0.205  max=0.604  mean=0.406
Unknown: n=10  min=0.016  max=0.104  mean=0.058

--- At CONFIRMED_THRESHOLD = 0.62 ---
False Acceptance Rate (FAR): 0.000
False Rejection Rate (FRR): 0.600

--- Threshold sweep (FAR over impostor+unknown, FRR over genuine) ---
  threshold    FAR     FRR
       0.30   0.600   0.000
       0.35   0.400   0.000
       0.40   0.300   0.000
       0.45   0.200   0.000
       0.50   0.167   0.000
       0.55   0.100   0.000
       0.60   0.033   0.200
       0.65   0.000   1.000
       0.70   0.000   1.000
       0.75   0.000   1.000
       0.80   0.000   1.000

--- Scores in current Uncertain band (0.45 <= score < 0.62) ---
  Genuine: 3 of 5
  Impostor: 6 of 20
  Unknown: 0 of 10
```

"Genuine" and "Impostor" here are the synthetic P1–P5; "Unknown" are the two real selfies. The 3 genuine scores
below 0.62 all land in **Uncertain**, not Unknown — the three-tier policy asks for another photo instead of rejecting.

### 2. Genuine identification (true identity enrolled)

| Photo | Condition | Predicted | Similarity | Decision |
|---|---|---|---|---|
| RealB `real_B1` / `real_B2` | exact enrollment photo (real) | RealB | 0.9547 / 0.9547 | Confirmed |
| SynthA `A1` | exact enrollment photo | SynthA | 0.9316 | Confirmed |
| SynthA `A3` | different photo | SynthA | 0.7808 | Confirmed |
| SynthA `A_test` | different pose / expression (hand on cheek) | SynthA | 0.7768 | Confirmed |
| P1 `p1_1` | exact enrollment photo | P1 | 0.9047 | Confirmed |
| P1 `p1_3` | different photo, outdoor / beach background | P1 | 0.6373 | Confirmed |
| P2 `p2_1` | exact enrollment photo | P2 | 0.9151 | Confirmed |
| P2 `p2_2` | different photo, **head turned, outdoor** | — | **0.4220** | **Unknown (false reject)** |

Both real enrollment photos score exactly 0.9547 by construction: the reference is the normalised mean of the two
unit vectors, so each photo sits at the same angle from it.

### 3. Unknown people (never enrolled)

| Photo | Who | Best match | Similarity | Decision |
|---|---|---|---|---|
| `real_B1`, `real_B2` | **real** person, before being enrolled | — | 0.0797, 0.0643 | Unknown |
| `aug_B1` dark / bright / rot15 / mirror | real, variants | — | 0.1001 / 0.0960 / 0.1023 / 0.0840 | Unknown |
| Gemini woman front / three-quarter / profile | synthetic | — | 0.1164 / 0.1335 / 0.1913 | Unknown |
| P4 ×3 (P2 look-alike) | synthetic | P2 / — / P2 | 0.4965 / 0.3518 / 0.4769 | Uncertain / Unknown / Uncertain |
| P3 ×3 (P1 look-alike) | synthetic | P1 | 0.5387 / **0.6215** / 0.4684 | Uncertain / **Confirmed** / Uncertain |
| P5 ×3 (P1 look-alike) | synthetic | P1 | **0.6201** / **0.6234** / 0.5181 | **Confirmed** / **Confirmed** / Uncertain |
| "B" (reuses A's face) | synthetic | SynthA | **0.7308** | **Confirmed** |

### 4. Impostor pairs (every photo vs every wrong reference)

160 impostor pairs; maximum 0.7308. At the default threshold (0.62) there are **4 false accepts**, all involving
synthetic look-alike identities:

| Impostor pair | Similarity |
|---|---|
| "B" vs SynthA | 0.7308 |
| P5 `p5_2` vs P1 | 0.6234 |
| P3 `p3_2` vs P1 | 0.6215 |
| P5 `p5_1` vs P1 | 0.6201 |

No confusion involves the real face: every other face scores **at most 0.121** against the real person's reference,
and the real person's photos score at most 0.102 against every synthetic reference. Photos of different *enrolled*
synthetic people score 0.17–0.48 against each other's references — Unknown or Uncertain, never Confirmed.

### 5. Pose / lighting / occlusion robustness (real face — variants of the enrollment photos)

| Condition | Similarity (from B1 / from B2) | Decision |
|---|---|---|
| Darker (−60% brightness) | 0.9434 / 0.9361 | Confirmed |
| Brighter (+70%) | 0.9193 / 0.9235 | Confirmed |
| Low contrast | 0.9409 / 0.9287 | Confirmed |
| Blur σ=3 | 0.9481 / 0.9368 | Confirmed |
| Heavy blur σ=10 | 0.8408 / 0.7933 | Confirmed |
| Mirrored (left/right) | 0.9399 / 0.9318 | Confirmed |
| Rotated 15° / 35° | 0.9478, 0.9455 / 0.9420, 0.9410 | Confirmed |
| Rotated 90° | 0.8821 / 0.8418 | Confirmed |
| Low resolution (1/10 size) | 0.9310 / 0.9113 | Confirmed |
| Eyes covered | 0.7784 / 0.7847 | Confirmed |
| Mouth covered | 0.7827 / 0.7733 | Confirmed |
| Tight face crop (no margin) | — | **No face detected** |

Independent photos with different conditions were only available for synthetic people (section 2): a different
expression (0.7768), a different background (0.6373) and a large head turn outdoors (0.4220 → Unknown).
**Not tested (no suitable photos):** real photos in other locations/lighting, real large head turns, real glasses.

### 6. Threshold sweep (identification decisions; Uncertain lower bound fixed at 0.45)

| Confirmed threshold | Genuine, different photo (n=4): accepted / rejected / Uncertain | Genuine exact + real variants (n=32): accepted | Unenrolled synthetic (n=13): false accepts / Uncertain | Real unknown (n=6): false accepts | Impostor pairs ≥ threshold (of 160) |
|---|---|---|---|---|---|
| 0.50 | 3 / 1 / 0 | 32 | 6 / 3 | 0 | 7 |
| 0.55 | 3 / 1 / 0 | 32 | 4 / 5 | 0 | 4 |
| 0.60 | 3 / 1 / 0 | 32 | 4 / 5 | 0 | 4 |
| **0.62** | **3 / 1 / 0** | **32** | **4 / 5** | **0** | **4** |
| 0.65 | 2 / 2 / 1 | 32 | 1 / 8 | 0 | 1 |
| 0.70 | 2 / 2 / 1 | 32 | 1 / 8 | 0 | 1 |

- 0.60 and 0.62 behave identically on this data.
- 0.65+ removes the synthetic P3/P5 look-alike false accepts but also rejects P1's genuine 0.6373.
- 0.50 adds two more synthetic false accepts.
- For the real face the gap between genuine (≥ 0.77) and unknown (≤ 0.13) is so large that every threshold from
  0.50 to 0.70 makes the same decisions.

### 7. Failure cases (observed in the running app)

| Case | Observed behaviour |
|---|---|
| No face (real background-only crop) | "No face detected. Please upload a clearer image." — no attempt used |
| Blank image | Same message — no attempt used |
| Two faces (composite) | Two independent results: P2 0.89, P1 0.89 |
| Five faces (synthetic composite) | Five results (all SynthA: 0.9262, 0.7196, 0.9116, 0.7747, 0.7589) |
| Unsupported type (`.txt`) | Blocked in the browser ("Only JPG or PNG images are supported"); API returns 400 |
| Empty enrollment database | "No enrolled identity matched", similarity 0.00 |
| Unknown person | Unknown with an "Enroll this person" action (never auto-enrolled) |
| Heavy blur (σ=10) | Still recognised (0.84 / 0.79) |
| Tight face crop | No face detected — the detector needs some context around the face |
| Enrollment with no-face / multi-face photos | Flagged per photo ("No face found", "Multiple faces"); Save stays disabled until 2 usable photos |

### 8. Retry behaviour (verified in both Next.js and Streamlit)

- Confirmed and Unknown results do not use an attempt; neither does a photo with no face.
- An Uncertain result uses exactly one attempt per **new** upload; a re-render / plain rerun does not.
- The third Uncertain upload locks the flow ("Identity could not be confidently verified. Maximum attempts reached (3).").
- "Try again" resets the counter, and so does a later Confirmed result. A full page reload resets it (never increments).

### 9. Evaluation threshold slider (verified end to end)

The slider sends `confirmed_threshold` with the identify request; FastAPI passes it to `best_match`. Same photo
(P5 held-out photo, similarity 0.5804 against P5):

| Threshold sent | Applied by backend | Decision |
|---|---|---|
| none (slider untouched) | 0.62 | Uncertain |
| 0.55 | 0.55 | Confirmed |
| 0.58 | 0.58 | Confirmed |
| 0.59 | 0.59 | Uncertain |
| 0.62 | 0.62 | Uncertain |
| 0.90 | 0.90 | Uncertain |

A face at similarity 0.07 stays Unknown at both 0.45 and 0.90 (the lower bound never moves). Out-of-range values
(0.30, 0.95) are rejected with HTTP 400. `src/config.py` was unchanged (identical SHA-256 before and after).

## Threshold selection reasoning

`CONFIRMED_THRESHOLD = 0.62` and `UNCERTAIN_LOWER_BOUND = 0.45` are **kept unchanged** as a deliberately
conservative operating point:

- **LFW (50 real people):** 0.62 gave 90.2% confident identification with **zero wrong-person confirmations and zero
  strangers confirmed**; the other 9.8% went to Uncertain (retry), none to Unknown.
- **Lowering it was considered and rejected.** 0.55 would raise LFW confident identification to 97.7%, but on the
  synthetic look-alike data thresholds below 0.62 produced wrong-person confirmations. The design choice is
  *borderline → Uncertain → ask for another photo*, not *borderline → confidently name someone*.
- **Using 3 enrollment photos is the better lever:** 90.2% → 95.9% confident identification at the same 0.62, with no
  change to the model or threshold (section 0). The Enroll page recommends 3 photos.
- **The 0.45 lower bound stays.** 0.40 was considered: the weakest LFW genuine score was 0.456 (just above 0.45) and
  the strongest stranger 0.313, and on the synthetic set it would move P2's head-turned photo (0.4220) from Unknown to
  Uncertain. That is one borderline case and a thin LFW margin — not enough evidence to re-tune the bound, so the
  evaluated value is kept. It is the first candidate to revisit with a larger real dataset.
- The remaining false accepts at 0.62 come from synthetic identities that are duplicate faces; the Uncertain band
  still catches most look-alikes.

## Observations & limitations

- Measured on 50 real LFW people (mostly frontal press photos, not look-alikes) plus one real person's selfies —
  not yet on phone/webcam photos, look-alikes, twins or relatives.
- Large head turns are the clearest genuine failure (0.4220 → Unknown). Enrolling an angled photo would likely help.
- Tightly cropped faces are not detected.
- Look-alike faces are the main false-accept risk; the Uncertain band catches many but not all of them.
- No liveness / anti-spoofing: a photo of an enrolled person would be accepted.
