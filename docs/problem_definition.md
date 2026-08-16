# Problem Definition and Task Framing

Stage 0 of `ML_WORKFLOW.md`. This document is written before any code, any data and any
scaffold. Every later stage must be traceable back to it. If a mid-project finding
changes the framing, this document is updated first, then the code.

Status: settled, 2026-08-16.

---

## 1. What problem are we solving?

An online gift retailer sells to a base of business customers who order irregularly.
There is no subscription and no contract, so a customer never announces that they have
left. They simply stop ordering, and by the time anyone notices, the relationship is
usually gone.

**We want to know, for each customer on the books today, whether they will place another
order in the next 90 days.**

Stated for the person who will use it: the retention team needs to know which of their
existing customers are about to go quiet, while there is still time to do something
about it.

## 2. Why are we solving it?

The output feeds one specific decision: **who receives a win-back offer in the next
retention campaign.**

- **Consumer:** the retention marketing team.
- **Action:** customers whose predicted probability of repeat purchase falls below a
  chosen threshold are placed in the campaign list and sent a discount or re-engagement
  offer. Everyone else is left alone.
- **What changes because the model exists:** today the team either contacts everyone
  (expensive, and it discounts customers who were going to buy anyway) or applies a
  manual RFM rule of thumb. The model replaces the rule of thumb with a ranked, calibrated
  list, and gives an explicit, costed answer to "how far down the list do we go?"

If the campaign list would be the same with or without the model, the project has
delivered nothing. That comparison is an explicit exit criterion in Stage 8.

## 3. What does success look like?

### Business success

Fewer customers lost to silent churn, at a defensible campaign cost. Concretely, the
model succeeds if the retention team can reach a materially larger share of the customers
who were about to churn, without inflating the number of discounts handed to customers
who would have bought anyway.

The two errors do not cost the same, and the direction matters:

| Situation | Reality | We predict | Action taken | What it costs |
|---|---|---|---|---|
| Missed churner | Will not repeat | Will repeat | No offer sent | The recoverable value of that customer. **Expensive.** |
| Wasted offer | Will repeat | Will not repeat | Offer sent | The discount, given to someone who needed no incentive. **Cheap.** |
| Correct save attempt | Will not repeat | Will not repeat | Offer sent | Discount cost, but with a real chance of retention. Intended spend. |
| Correct no-contact | Will repeat | Will repeat | No offer sent | Nothing. |

**Missing a churner costs more than wasting an offer.** The decision layer must therefore
lean toward recall on the non-repeat class, and the exact operating point comes from the
cost model below rather than from a default 0.5 threshold.

### The cost model

Let:

- `V` = expected forward 90-day margin from a customer who stays.
- `e` = probability that a win-back offer actually converts a would-be churner
  (offer effectiveness). No campaign saves everyone.
- `d` = cost of the discount when redeemed.

Then the recoverable value at stake per churner is `e * V`, and:

- Cost of a **missed churner** = `e * V` (the benefit forgone).
- Cost of a **wasted offer** = `d`.

Only the ratio matters for choosing a threshold. We normalize `e * V = 1.0` and assume
`d = 0.15`, that is, **the discount costs about 15 percent of the value it can recover**,
giving a working ratio of

```
COST_MISSED_CHURNER : COST_WASTED_OFFER  =  1.0 : 0.15
```

This ratio is an assumption, not a measurement. Stage 8 reports the sensitivity: at what
ratio does the chosen threshold move materially, and at what ratio would the model
choice itself flip. If the retention team later supplies real campaign economics, the
ratio changes in `src/config.py` and only the threshold needs recomputing, not the model.

### Technical success

- **Primary metric:** ROC-AUC, measured out of fold on the training set. It is chosen for
  model *selection* precisely because it is threshold independent, which keeps the
  selection decision separate from the operating-point decision.
- **Secondary metrics:** PR-AUC, and at the chosen operating point, precision and recall
  **on the non-repeat class**, because that is the class the campaign acts on.
- **Minimum acceptable bar, all of which must hold:**
  1. Out-of-fold ROC-AUC of at least **0.75**.
  2. The model beats the RFM quintile heuristic on out-of-fold ROC-AUC by a margin larger
     than the fold-to-fold standard deviation. Beating it by less than the noise is not
     beating it.
  3. At the chosen threshold, the model's expected campaign cost is at least **20 percent
     below the best trivial policy**, where the trivial policies are: contact nobody
     (cost = `COST_MISSED_CHURNER` times the number of churners), contact everybody
     (cost = `COST_WASTED_OFFER` times the number of repeaters), and the RFM heuristic.

Criterion 3 is the one that actually matters to the business. A model can win on AUC and
still fail it.

---

## 4. Task type

**Binary classification, one row per customer.**

| Property | Choice |
|---|---|
| Grain | One row per customer, aggregated from the transaction log |
| Positive class | `y = 1` if the customer places at least one non-cancelled order in the prediction window |
| Negative class | `y = 0` otherwise (this is the class the campaign targets) |
| Output | `P(repeat within 90 days)`. Churn risk is reported as `1 - P(repeat)` |

A note on class polarity, because it is a live source of confusion downstream: the model
predicts **repeat**, matching the project name and making `predict_proba[:, 1]` mean the
intuitive thing. The business action fires on the **negative** class. So a false positive
in standard label terms (predicted repeat, actually did not) is the *expensive* missed
churner, and a false negative is the *cheap* wasted offer. This inversion is deliberate
and is why the cost constants in `src/config.py` are named
`COST_MISSED_CHURNER` and `COST_WASTED_OFFER` by their business meaning rather than
`COST_FP` and `COST_FN`.

### Why not the alternatives

- **Not a time series forecast.** We do not need a volume of orders per week. We need a
  per-customer decision, and the customer is the unit the campaign acts on.
- **Not a regression on next-order value.** Useful later for prioritizing within the
  campaign list, but the first decision is binary: contact or do not contact.
- **Not survival analysis**, which would model time-to-next-order and is the more
  sophisticated framing. It is deliberately deferred: the 90-day binary version answers
  the actual question, and Principle 2 of the workflow favours establishing an honest
  simple result before adding machinery.

---

## 5. Temporal design

This is the single most important design decision in the project, because it is where
leakage would enter.

```
OBSERVATION WINDOW (features)              PREDICTION WINDOW (label)
2009-12-01 ───────────────────► 2011-09-11 ──────────────────► 2011-12-10
        about 21 months of history            90 days
                                    ▲
                                 CUTOFF
```

| Parameter | Value |
|---|---|
| `OBSERVATION_START` | 2009-12-01 (first transaction in the dataset) |
| `CUTOFF_DATE` | 2011-09-11 |
| `HORIZON_DAYS` | 90 |
| Prediction window | `[2011-09-11, 2011-12-10)` |

The cutoff is set so that `CUTOFF_DATE + HORIZON_DAYS` clears the last timestamp in the
dataset, keeping the prediction window fully populated. Confirmed in Stage 2: the data
runs to 2011-12-09 12:50:00, and an exclusive bound of 2011-12-10 loses nothing. See
`docs/data_provenance.md` for what the earlier one-day-short boundary would have cost.

**Population:** customers with a non-null `Customer ID` and at least one non-cancelled
invoice strictly before the cutoff. A customer who first appears only in the prediction
window cannot be scored, because at cutoff time they did not exist to us.

**Rules that follow, and are non-negotiable:**

1. No transaction dated on or after `CUTOFF_DATE` may influence any feature. Features come
   from the observation window only; the label comes from the prediction window only.
2. Recency is measured against `CUTOFF_DATE`, never against the dataset's own maximum
   date. Using `max(invoice_date)` is the classic silent leak in this dataset: it imports
   knowledge of the future into a feature that looks innocent.
3. Every rolling window feature (`orders_last_90d` and similar) is anchored to
   `CUTOFF_DATE`.

---

## 6. Validation strategy

Each customer contributes exactly one row, built from one snapshot at one cutoff. There
is no time axis remaining inside the customer table and no entity repeats across rows, so
neither a temporal split nor a group-aware split is required at this grain.

- **Split:** random, stratified on `y`, 80/20, `random_state = 42`. Performed **once**,
  saved to `data/processed/train.parquet` and `test.parquet`. Every later stage loads
  those files. No stage re-splits.
- **Tuning:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training
  file only. The same fold object is reused across every model family so comparisons are
  fair.
- **Model selection happens on out-of-fold CV metrics, not on the test table.** Picking the
  winner by comparing test scores across candidates is selection on the test set and
  inflates the reported number, even when every other step was clean.
- **Threshold selection happens on out-of-fold training probabilities**, then the chosen
  threshold is applied to the test set exactly once.
- The test set is touched once per model, for final confirmation of a decision already
  made.

---

## 7. Baselines

Three reference points, in increasing order of seriousness. The model must clear all
three.

1. **Trivial:** predict the majority class for everyone. Establishes the floor and the
   base rate.
2. **Incumbent:** an RFM quintile heuristic, the rule a marketing team would apply with a
   spreadsheet and no model (score each customer 1 to 5 on recency, frequency and
   monetary; target the low scorers). This is the real competitor. Beating the majority
   class proves nothing; beating the incumbent proves the project earned its cost.
3. **Interpretable model:** Logistic Regression inside a scaling pipeline. Provides the
   first coefficient-level story of what drives repeat purchase, and a sanity check on
   the features before any tree ensemble hides problems behind flexibility.

The candidate roster for Stage 7 is fixed now, before any scores exist: **Logistic
Regression, XGBoost and LightGBM**. Random Forest was dropped for computation cost, and
loses nothing the two boosters do not already cover. Fixing the roster in advance keeps
the comparison honest; picking candidates after seeing results is selection by another
name. See decision 7.1 in `CLAUDE.md`.

---

## 8. Decision layer

Required, and planned for now rather than bolted on later.

- The threshold is a **tuned parameter with a business objective**, not the sklearn
  default of 0.5.
- Objective: minimize expected campaign cost,
  `COST_MISSED_CHURNER * (missed churners) + COST_WASTED_OFFER * (wasted offers)`,
  swept over candidate thresholds on out-of-fold training probabilities.
- **One knob per problem (Principle 4).** Threshold tuning is our chosen mechanism for
  handling class imbalance and asymmetric cost. Therefore **no `class_weight="balanced"`,
  no SMOTE and no resampling anywhere in the project.** Stacking both would be two
  mechanisms solving the same problem, with neither one's contribution measurable.
- F2 on the non-repeat class is reported as a cross-check on the cost-minimizing
  threshold. It is a check, not a second tuning knob, and it never overrides the cost
  objective.
- The threshold is persisted in `models/model_selection.json` and then in
  `models/model_metadata.json`, and is loaded by the serving layer. It is never hardcoded
  in a notebook or in the API.

---

## 9. Data

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository, dataset 502, "Online Retail II" |
| Contents | Transaction log of a UK-based online gift retailer |
| Period | 2009-12-01 to 2011-12-09 |
| Volume | About 1.07 million line items across two Excel sheets |
| Grain | One row per invoice line item |
| Columns | `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country` |

Known characteristics to confirm and quantify in Stage 3, not to assume:

- A substantial share of rows carry a null `Customer ID` and cannot be attributed to a
  customer. They must leave the modeling population, and the revenue they represent has
  to be quantified so the exclusion is a sized decision rather than a silent one.
- Invoices prefixed `C` are cancellations and carry negative quantities.
- `StockCode` includes non-product entries (postage, manual adjustments, bank charges,
  samples, test rows, gift vouchers) that are not purchases and must be classified
  explicitly.
- `Country` is dominated by the United Kingdom, which is what forces a pinned vocabulary
  with an `Other` bucket rather than letting an encoder infer categories from whatever
  batch it sees.

---

## 10. Constraints, assumptions and known limitations

Recorded now so that no reviewer has to discover them later.

1. **Seasonality is confounded and cannot be avoided.** The 90-day prediction window
   (September to December) covers the Christmas peak for a gift retailer, so the base
   rate is seasonally inflated. The dataset ends 2011-12-09, so no cutoff choice escapes
   this. Consequence: the absolute probabilities are specific to a pre-Christmas window,
   and deploying against an off-peak quarter would require recalibration. This is a
   documented limitation, not a defect to engineer around.
2. **The cost ratio 1 : 0.15 is assumed, not measured.** Sensitivity analysis in Stage 8
   is what makes this honest.
3. **Offer effectiveness `e` is folded into the normalization** rather than estimated. We
   have no campaign response data, so `e` cannot be identified from this dataset. A real
   deployment would need a holdout campaign to measure it.
4. **The data is a historical snapshot from 2009 to 2011.** Nothing here is a claim about
   a live business today.
5. **Single cutoff, single snapshot.** Multiple stacked cutoffs would yield more training
   rows, at the cost of the same customer appearing several times and requiring
   group-aware splitting. Deferred deliberately, and noted as the first thing to try if
   the data volume proves limiting.
6. **We model whether a customer returns, not what they will spend.** Campaign
   prioritization by value is out of scope.

## 11. Out of scope

Product recommendation, next-order value regression, customer lifetime value modeling,
market basket analysis, and any campaign-response uplift model. Each is a reasonable
follow-on and none is needed to answer the question this project asks.

---

## Stage 0 exit check

| Requirement | Status |
|---|---|
| What problem are we solving, in writing | Section 1 |
| Why, and what decision the output feeds | Section 2 |
| Business success, including cost of each error direction | Section 3 |
| Technical success: metric plus minimum acceptable value | Section 3 |
| Task type identified explicitly | Section 4 |
| Primary metric named before any modeling | ROC-AUC, out of fold |
| Validation strategy named | Section 6 |
| Trivial baseline named | Majority class, plus RFM incumbent |
| Decision layer need identified | Section 8 |

All satisfied. Stage 1 (Project Scaffold) may begin.
