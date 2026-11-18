# 1. Search as composable stages, a design as config

Status: accepted

## Context
The posting asks for multistage composable search: query understanding, retrieval, personalised re-ranking. Teams
ship small changes weekly, and each needs comparing against the others on the same traffic.

## Decision
One `Stack.search` with five stages (understand, filter, retrieve, fuse, rerank). A design is a TOML table that
switches and parameterises stages (`config/designs.toml`). The same process serves every design; the service takes
the design by name.

## Consequences
Adding a variant is a config change, so the offline-to-online study grows for free. A new stage must default to off
so every existing design stays byte-identical, which same-luck forks make checkable.
