# The Data and What We Found, In Plain Language

A non-technical walkthrough of the dataset and the Stage 3 (EDA) and Stage 4 (cleaning)
findings. Written for a quick read, not as a technical reference. See
`docs/data_provenance.md` and `docs/problem_definition.md` for the technical version.

---

## What the data actually is

It's the sales record of a UK online gift shop, from December 2009 to December 2011 —
about two years. Every row is one line on a receipt: "customer X bought 12 of item Y on
this date for this price." Over a million rows like that.

## Who's in the data

**About 6,000 customers.** But here's the first surprise: **23 out of every 100 rows have
no customer attached to them.** No name, no ID, nothing. That's nearly a quarter of all
sales just... unassigned. We can't build a "will this customer come back" prediction for
someone we can't identify, so those rows get set aside. The good news: they're mostly
small purchases, only about 14% of the actual money.

## The receipts aren't all real purchases

Not every row is a customer buying a product. Digging through it, we found several kinds
of "fake" rows mixed in:

- **Shipping charges** ("POST", "DOTCOM POSTAGE") — that's a delivery fee, not a product
- **Bank fees, Amazon fees** — accounting stuff
- **"Adjust bad debt" entries** — just 6 rows, but they represented a huge chunk of money
  (someone writing off money the shop never collected)
- **Test rows** — literally labeled "This is a test product"
- **Gift vouchers** — that's a payment method, not a thing someone bought
- **Warehouse write-offs** — damaged, missing, or "thrown away" stock. Interesting
  finding: these all had a price of £0 and no customer attached, so they're not customer
  returns, they're just the warehouse cleaning its books

All of that got removed, because it would confuse the model into thinking postage or a
warehouse mistake is a "purchase."

## The tricky part: cancellations vs. returns

Some rows have a negative quantity, meaning something was given back. We first assumed
"negative quantity = a return." That turned out to be wrong. The real signal is a "C" at
the start of the receipt number — that's the official cancellation marker. Some
negative-quantity rows had no "C" and turned out to be those warehouse write-offs
mentioned above, not customer returns. So we cleaned by the "C" marker, not by the
negative number.

## Two sneaky mistakes we caught along the way

**Mistake 1 — cancellations were getting erased by accident.** When the computer first
read the file, it looked at the receipt numbers, saw they were mostly plain numbers, and
decided "this whole column is just numbers." But cancellation receipts look like
"C508404" — they have a letter in front. Since the computer had already decided "numbers
only," it couldn't fit those in and just threw them away — 19,500 of them, silently. No
error message, nothing looked broken. We only caught it because we double-checked the
data type of that column against what we knew should be there. If we hadn't caught it,
the model would have had no idea which customers return things.

**Mistake 2 — we were about to lose the last day and a half of sales.** The way we sliced
"past data" vs. "future data" for the model had an off-by-one problem — it cut off a few
hours too early, right at the very end of the dataset. That would've silently thrown out
real transactions and, worse, wrongly labeled 3 customers as "didn't come back" when they
actually did. Fixed by moving the cutoff line by one day.

## What we ended up with

Started at 1,067,371 rows. After removing the shipping/fees/test rows/write-offs and
non-customer rows: **1,055,289 real, clean transaction rows**, about 6,000 real
customers.

## The most useful thing we learned: what actually predicts "will they come back?"

We tested three classic signals:

1. **How recently did they last buy something?** — This one matters a lot. Customers who
   come back bought something about 2 months ago on average; customers who don't come
   back last bought about 9-10 months ago.
2. **How often do they buy?** — Matters, but less.
3. **How much have they spent total?** — Matters, but even less than expected, because a
   handful of huge spenders skew the raw numbers.
4. **How long have they been a customer at all?** — Turned out to barely matter at all.
   Someone who's been shopping there for 2 years isn't meaningfully more likely to return
   than someone who joined 6 months ago. What matters is *recent* activity, not *tenure*.

One counterintuitive finding: **customers who've cancelled an order are actually more
likely to come back**, not less. That seems backwards until you realize — you can only
cancel an order if you've placed several orders in the first place. So "has cancelled
something" is really just a side-effect of "buys a lot," not a sign of an unhappy
customer.

## The bar we have to beat

We also built the "dumb but effective" version a marketing person could do by hand in a
spreadsheet: rank customers 1-5 on recency, frequency, and spending, add up the score.
That simple rule alone already sorts customers from a 7% chance of returning up to a 91%
chance. That's genuinely good. So our actual machine learning model has real work to do
to justify itself — it has to clearly beat that spreadsheet trick, not just an outcome of
"flip a coin."
