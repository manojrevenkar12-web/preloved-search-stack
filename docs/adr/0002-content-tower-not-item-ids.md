# 2. A content tower, not item-ID embeddings

Status: accepted

## Context
Every pre-loved listing is one of a kind: it is bought once and leaves. An embedding per item ID can only learn from
the clicks that item got, and most items sell or expire before they get any.

## Decision
The item tower embeds what the listing says (title words and character trigrams, the seller's attributes, a price
band), hashed into 2^15 buckets. The query tower embeds the query text and the slots read from it. Trained in PyTorch
with in-batch softmax on (query, clicked item) pairs, purchases weighted three times; served from NumPy.

## Consequences
New listings are retrievable on the day they are listed. The ID-tower ablation (`id_tower`) measures the cost of the
usual recommender default; see the README. Nicknames such as "loubs" and "slp", which no dictionary in the stack
contains, are learned from clicks.
