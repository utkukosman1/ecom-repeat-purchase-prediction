# Model Selection Decision

Stage 8 of `ML_WORKFLOW.md`, the final stage of this project. Full evidence in
`notebooks/06_comparison_and_threshold.ipynb`. This document is what a reviewer would
ask for: the numbers behind the call, and why the call is what it is.

## The decision

**Do not deploy a model. Continue using the RFM heuristic.**

XGBoost is the best-performing model found, and it is saved along with a tuned
threshold in `models/model_selection.json` for the record. But it does not clear the
bar this project set for justifying a model over the existing rule.

## The numbers

Out-of-fold ROC-AUC, all four candidates scored the same way (leakage-free, on training
data the model or rule never saw while being fit):

| Candidate | OOF ROC-AUC | Beats RFM by more than its own noise |
|---|---|---|
| RFM heuristic | 0.7811 | not applicable |
| Logistic Regression | 0.7942 | yes |
| LightGBM | 0.7979 | yes |
| XGBoost | 0.8018 | yes |

Cost per customer at each candidate's own best threshold, applied to the untouched test
set exactly once:

| Policy | Cost per customer |
|---|---|
| Contact nobody | 0.5646 |
| Contact everybody | 0.0653 |
| RFM heuristic | 0.0613 |
| XGBoost | 0.0589 |

Cost reduction, XGBoost versus the best of the three trivial policies (RFM): **3.9
percent.** The bar was 20 percent.

## Why the ROC-AUC win did not turn into a cost win

Both of these are true at once, and neither cancels the other:

1. XGBoost really does rank customers better than RFM. The gap (0.021) is bigger than
   the fold-to-fold noise (0.011), so it is not a coincidence of one split.
2. That better ranking barely matters here, because the cost of missing a churner is
   6.67 times the cost of a wasted discount offer. At that ratio, the cheapest policy
   for every candidate, RFM included, is to offer nearly everyone a discount. XGBoost's
   best threshold sends offers to 93.5 percent of customers. Once nearly everyone is
   already being offered something, a model with sharper ranking has very little room
   left to save money: it can only pick off the confident non-churners at the very top
   of the distribution, and there just are not many of them.

A model earns its complexity by changing what the business does. Here it barely changes
the decision. Almost everyone gets the offer either way.

## What would change this conclusion

- **A cheaper wasted-offer cost, or a larger dataset.** The current cost ratio and the
  roughly 4,200 training rows both compress how much room a model has to separate
  itself from the rule. Either one changing could move the answer.
- **A cost ratio the retention team can measure**, rather than the 1.0 : 0.15 assumed in
  `docs/problem_definition.md`. If missing a churner is cheaper than assumed, or a
  wasted offer more expensive, the optimal threshold moves away from "offer almost
  everyone" and a model's finer ranking would matter more.
- **More data.** XGBoost's fold-to-fold standard deviation (0.011) is a meaningful share
  of its total advantage. A larger customer base would tighten that and could either
  confirm or erase the gap.

## What is not being recommended

Not "the project failed." The honest finding is that a spreadsheet rule the retention
team can already run is not worth replacing with a trained model, on this data, at this
cost ratio, right now. That is a legitimate outcome, not a consolation prize: Stage 0
deliberately left this door open rather than assuming a model would win.
