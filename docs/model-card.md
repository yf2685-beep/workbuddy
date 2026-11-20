# Data and model card

## Intended use

An educational portfolio prototype for review analytics, chronological evaluation and interactive clothing-fit classification. Predictions are not a validated retail sizing service. The UI recommends the observed size code with the largest modeled probability of `fit`; this ranking has not been directly evaluated with counterfactual fit labels.

## Dataset and cleaning

- Source: [RentTheRunway / UCSD Clothing Fit](https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit), Misra, Wan & McAuley, RecSys 2018.
- Input SHA-256: `c24b32298c8732521fa19fcd216401f609f170bbfccf3359c038f803c711c7ae`.
- 192,544 original JSONL rows; 189 exact duplicate objects removed; 192,355 retained.
- 675 retained rows have missing/invalid height; 29,961 have missing/invalid weight.
- Feet/inches converted to cm, pounds to kg. Invalid/nonfinite measurements become missing. Rows missing a valid date, target label, size, item ID or user ID are rejected.
- Height defaults to 170 cm and weight to 70 kg in numeric features, with separate missingness indicators. These are fixed engineering defaults, not estimated population means. Missing body type/category become `unknown`.
- Size remains the dataset's original numeric code; no universal brand conversion is asserted.

Public artifacts contain fitted weights, metrics and per-item counts, not raw reviews or user IDs. The dataset page is publicly accessible; no explicit dataset license was found during setup. Refer to the source's applicable terms. The MIT code license does not apply to third-party data.

## Evaluation protocol

Sort by review date and choose approximately 70%/15%/15% cut points; keep each complete date in a single split. Training ends 2017-03-26, validation runs 2017-03-27 through 2017-08-29, and test runs 2017-08-30 through 2018-01-08. Exact counts, label frequencies, confusion matrices and per-class metrics are in [metrics.json](../reports/metrics.json).

The timestamp is the recorded review date, not a verified purchase date. This protocol isolates future reviews, but cannot prove all history was available before the target purchase.

Training features for a day are computed before adding any of that day's reviews. Validation/test histories are frozen at the training cutoff; neither split updates history. Feature dictionaries and scaling are fitted on training data only. Target review, target label, rating and user ID are excluded from each example's predictors. This is a time split, not a user-disjoint or item-disjoint cold-start evaluation.

## Features and estimators

| Family | Features |
| --- | --- |
| Metadata | Size code, height, weight, missing indicators, category and body type |
| Historical enhancement | Log item feedback count, smoothed item small/fit/large distribution, fraction of item reviews with a negative mention of each aspect |

Fit proportions use `(label_count + 3) / (history_count + 9)`. New items have a uniform prior and zero negative-aspect rates. Categorical features are one-hot encoded; unknown values at inference contribute no active training category. Numeric and one-hot features pass through `StandardScaler`.

Models compared:

1. Training majority-class predictor.
2. Metadata logistic regression: balanced class weights, `C=1`, `max_iter=500`.
3. Enhanced logistic regression: same settings plus historical features.
4. Enhanced sklearn MLP: `(64, 32)` ReLU layers, balanced sample weights, batch size 512, `alpha=0.001`, learning rate 0.001, at most 35 epochs.

Seed 42. MLP uses sklearn's random 10% internal training holdout for early stopping (`n_iter_no_change=5`), separate from the chronological outer validation set. The internal early-stopping split is not itself a temporal evaluation; only the outer validation/test support the reported chronological results. MLP stopped after 14 iterations in this run. This is not a PyTorch implementation or reproduction of the cited latent-factor model.

Among the fitted models, select highest outer validation Macro-F1; no final refit on train+validation. The selected historical logistic model achieves validation 0.4154 and test 0.3634 Macro-F1. The majority baseline scores 0.2811 test Macro-F1 and metadata logistic 0.2983.

Serialization exports feature names, scaler statistics, weights and biases to JSON. NumPy inference is checked against sklearn probabilities on 100 held-out rows per model (`atol=2e-5`). The demo retains aggregate history for only 250 high-volume training items; experimental metrics cover the full evaluation set, not just these items.

## Interpretation and limitations

- The aspect engine used in training is a sentence-level English rule baseline. A negative term can affect multiple aspects in the same sentence; negation, mixed sentiment and indirect language are not robustly handled. Human extraction accuracy is unknown.
- History enhancement changes both fit-label aggregates and text-derived features. The measured improvement cannot isolate the contribution of text without an additional ablation.
- Balanced class weights improve minority attention but lower total accuracy relative to always predicting `fit`. Macro-F1, accuracy and class-specific results should be read together.
- Probabilities are not calibrated. Size ranking is an illustrative use of classification probabilities, not independently validated recommendation performance.
- The dataset covers an older rental-fashion population and is not representative of every country, brand, gender or contemporary catalog. No subgroup fairness audit or online A/B test has been performed.
- One seed/run, no confidence intervals, no hyperparameter sweep, no ModCloth external evaluation. Lower test than validation performance signals limited generalization under this protocol.
- Item-level historical aggregates may be stale; production use would require a carefully evaluated update policy and stronger product/size information.

## Human annotation workflow (not yet completed)

Export a fixed random sample of up to 200 locally downloaded reviews:

```bash
workbuddy annotations --count 200
```

The generated `data/annotations/reviews.jsonl` is ignored by Git. Every row initially has `findings: null`. A human must replace this with a list of supported aspect/sentiment/evidence objects, or `[]` for a reviewed example with no relevant aspects:

```json
{
  "id": 0,
  "review": "The fabric is soft.",
  "findings": [
    {"aspect": "material", "sentiment": "positive", "evidence": "The fabric is soft."}
  ]
}
```

This is a synthetic annotation example, not a completed label. Allowed aspects: `sizing`, `fit`, `material`, `length`; sentiments: `positive`, `negative`, `neutral`. Quote a supporting substring exactly. Label each aspect according to its own meaning, not the sentence's overall tone. Mixed sentiment can produce more than one finding; omit unsupported aspects.

```bash
workbuddy evaluate-aspects data/annotations/reviews.jsonl
```

The command rejects incomplete labels and reports micro precision/recall/F1 over unique `(aspect, sentiment)` pairs per review for the rule baseline. It validates evidence but does not score span boundaries. An LLM benchmark, annotation agreement study and adjudication remain future work. Do not publish raw sampled reviews with annotations in this repository.
