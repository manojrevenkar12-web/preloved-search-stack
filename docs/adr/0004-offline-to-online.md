# 4. Judge offline metrics by whether they predict the online result

Status: accepted

## Context
The posting asks for an offline-to-online framework. An offline metric is only useful if, across the changes a team
actually considers, it orders them the way an A/B test would.

## Decision
Every design and variant is rolled out to all buyers in a fork of the same world (same buyers, searches and luck);
its purchase lift over today's search is the online truth. Each offline metric is scored by Kendall's tau against that
lift, by whether it picks the same winner, and by whether it gets the sign right. A/B tests (delta method, CUPED),
A/A false positives and interleaving efficiency are measured on the same world.

## Consequences
The study can say a popular metric is misleading, with a number. Its limit: one simulated marketplace, twelve
configurations; the held-out world is the check that the conclusions are not tuned to the dev seed.
