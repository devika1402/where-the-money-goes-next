# Glossary

Every term below is defined as this project uses it. Where a word has a wider meaning in the field,
the definition here is the narrow one the code implements.

## Problem

| Term | What it means here |
| --- | --- |
| money mule | A person whose bank account receives money that came from a crime and sends it onward. Most are recruited by an advert. |
| mule account | An account that received at least one payment the simulator marked as laundering, inside its label window. `src/definitions.py` states this rule once and every module imports it. |
| the simulator | A program that generated the dataset. It planted the laundering patterns by rule, so a model that finds them is partly finding the rule. |
| authorised push payment fraud | Fraud where the victim makes the payment, after a scammer talks them into it. Every check on the sending side passes. |
| variant | One of the two data files. `LI-Small` carries the published figures. `HI-Small` is the development file and has more mule accounts. |

## Windows and populations

| Term | What it means here |
| --- | --- |
| feature window | Four days of history that a score is computed from. |
| label window | Two days after the feature window. An account counts as a mule account in this window or in none. |
| leakage cutoff | Instant the feature window ends and the label window starts. No feature reads a payment at or after it. |
| scored population | Accounts that received at least one payment during the feature window. An account with no incoming payment has no features, so the pipeline cannot score it. |
| base rate | Share of the scored population that are mule accounts. Economics reports a different base rate for the daily alert queue, which counts an account once for each day it received money, so the two figures differ. |
| reachability ceiling | Share of the label window's mule accounts that fall inside the scored population. No model can pass it. |
| budget ceiling | Share of the label window's mule accounts that 400 alerts reach if every alert is correct. |

## Twenty account measures

Twenty measures are computed for each account in the scored population. All twenty read the
feature window only.

| Measure | What it counts |
| --- | --- |
| `in_degree` | Number of payments the account received. |
| `out_degree` | Number of payments it sent. |
| `unique_counterparties_in` | Number of separate accounts that paid it. |
| `unique_counterparties_out` | Number of separate accounts it paid. |
| `total_inflow` | Money it received, in euro. |
| `total_outflow` | Money it sent, in euro. |
| `max_inflow` | Its largest single incoming payment. |
| `max_outflow` | Its largest single outgoing payment. |
| `inflow_concentration` | Largest single sender's share of the money it received. |
| `outflow_concentration` | Largest single recipient's share of the money it sent. |
| `sender_diversity` | Separate senders divided by payments received. A value near 1 means almost every payment came from a different account. |
| `degree_asymmetry` | Payments in minus payments out, divided by the total of both. |
| `counterparty_asymmetry` | Same measure over separate counterparties. |
| `mean_amount_ratio` | Average outgoing payment divided by the average incoming payment. |
| `pass_through_ratio` | Money out divided by money in. Inflow counts the first three days of the feature window and outflow counts all four, so money that arrives late still has time to leave. It has no value when the account received under 100 EUR. |
| `median_hours_to_outflow` | Median hours from a payment arriving to the account's next outgoing payment. |
| `active_days` | Number of separate days the account sent or received anything. |
| `burstiness` | Busiest day's share of the account's payments. |
| `pagerank` | PageRank over the feature window payment graph. It rises for an account that busy accounts pay. |
| `reciprocity` | Share of the accounts it paid that also paid it. |

## Scoring and comparison

| Term | What it means here |
| --- | --- |
| scorer | One of the three ways this project ranks accounts: hand-written rules, logistic regression, and XGBoost. |
| percentile rank | Where an account falls in the population on one measure, from 0 to 1. |
| PR-AUC | Area under the precision-recall curve. It rises when mule accounts rank nearer the top. A scorer that has learned nothing scores the base rate. |
| bootstrap interval | A measure recomputed on 1,000 random re-draws of the same accounts. Its 2.5% and 97.5% columns give the range the middle 95% of those re-draws fall in. |
| paired difference | Two scorers measured on the same re-draw, then subtracted. Shared sampling variation cancels. |
| crosses zero | Interval on a paired difference includes zero, so the sign of the difference is not established. |
| early stopping | Training stops when the score on the validation window stops improving. |
| tree depth | How many splits deep one decision tree runs. This project allows 6. |
| gain-weighted feature importance | How much each measure reduced the training loss, as a share of the total. |
| `scale_pos_weight` | XGBoost setting for how much more one mule account counts than one clean account. Measured from the training labels. |
| mean degree | Average number of separate counterparties per account in the feature window graph. |
| local clustering | Whether an account's counterparties also pay each other. It is zero for an account in no triangle. |

## Budget and the money

| Term | What it means here |
| --- | --- |
| alert | An account the analyst team opens and works. |
| alert budget | Alerts a team can work in a day. This project assumes 200, which is 400 over a two-day label window. |
| threshold | Score of the last account inside the budget, fixed by the budget itself and not chosen by a person. |
| operating point | What the budget bought: alerts raised, catches made, threshold, and cost. |
| precision@k | Of the accounts alerted, the share that are mule accounts. Here the count is 400. |
| catch | An alert on an account that turns out to be a mule account. |
| recall | Share of the mule accounts in the day's queue that the alerts caught. |
| exposure | Money that arrived in an account during its label window, in euro. It prices a catch. |
| net EUR/day | Money recovered minus the cost of working the alerts, per day, under the four cost numbers. |
| break-even precision | Precision one more alert must reach to cover its own cost. |
| queue overflow policy | What happens to an account above the threshold that the day's capacity did not reach. `same_day` discards it. `rollover_max_3d` carries it for three days. |
| carried | Alerts a day spent on accounts that arrived on an earlier day. |

## Drift

| Term | What it means here |
| --- | --- |
| PSI | Population Stability Index. It compares how a measure is spread in the training window with how it is spread in a later window. |
| quantile bin | One of 10 equal-size bins cut from the training window. Those same edges apply unchanged to the later window. |
| empty bin | A bin holding nothing on one side. Its zero is replaced by 0.000001 before the logarithm, so the size of that bin's term comes from the replacement. |
| flag | A drift row with at least one empty bin. Read it as a warning that a bin emptied. |
| magnitude | A drift row with no empty bin. Read it as a distance. |
| KS | Largest gap between two cumulative distributions. It needs no bins and no reference period, and cannot say which part of the distribution moved. |
| degenerate feature | A measure that takes one value for almost every account. Every bin edge falls on that value, so PSI reads 0.0000 and cannot see the measure at all. |
