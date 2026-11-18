# The held-out world

`config/heldout.toml`: the same marketplace rules as dev with a different seed (977), so different buyers, sellers,
items and luck, and a messier catalogue: more missing attributes (20% instead of 15%), more wrong colours, more typos
and nicknames, more English-speaking buyers.

Every choice (features, designs, parameters, the gate, the metrics in the study) was fixed on dev first. Then
`make heldout` ran history, training and evaluation once and wrote `reports/heldout.lock` with the time and commit.
`scripts/heldout.py` refuses to run while the lock exists. The README shows the held-out numbers beside dev's as they
came out.
