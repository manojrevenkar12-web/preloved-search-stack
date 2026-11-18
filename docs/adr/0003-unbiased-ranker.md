# 3. Train the ranker against position bias, measured from randomised traffic

Status: accepted

## Context
Clicks come from pages the old ranking ordered. A ranker trained on raw clicks partly learns "what the old ranking put
first".

## Decision
5% of searches get their top 10 shuffled. The click rate by position on that traffic, fitted with a power law, is the
examination probability. The listwise loss weights each click by its inverse (IPS). The `biased_ranker` ablation trains
the same model without the weights.

## Consequences
The estimate is checked against the world's true examination curve in the README. Shuffling costs a little on 5% of
searches; the report shows the purchases lost.
