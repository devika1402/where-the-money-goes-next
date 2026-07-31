# Where the Money Goes Next: which receiving account to investigate first

When someone is talked into transferring their own money to a scammer, the bank cannot catch it by
looking at the payment. The customer made the transfer themselves, from their own device, after
passing their own security checks, to an account they typed in themselves. Everything a fraud
system watches on the sending side looks correct, because the deception happened in a phone call
before any of those checks ran.

![Why do we score the receiving account? Three stages read top to bottom: every sender-side check passes, since device, location and login all look normal. Money arrives at the account from several strangers with no shared history. The same money leaves again within 24 hours, to new accounts.](docs/diagrams/second-hop.svg)

So the account at the other end is the one to look at. Money taken this way has to move on quickly,
through an account someone was recruited to lend out. That account is a mule account, and it looks
different from an ordinary one even when every individual payment looks fine. This project scores
the receiving account. Out of
260,459 accounts that took money in over the previous four days, it works out which 400 an analyst
team should open first. A fixed daily budget sets that number, and the pipeline prices what being
wrong in each direction costs.

## Documentation map

| Document | What it covers |
| --- | --- |
| README.md (this file) | What this is, what it found, and how to run it |
| [docs/results.md](docs/results.md) | The full comparison: rules against models, the queue policy, and drift |
| [docs/glossary.md](docs/glossary.md) | Every term these pages use, defined as this project uses it |
| [reports/report.md](reports/report.md) | The generated results document, assembled from the metrics files without recomputing anything |
| [reports/briefs/](reports/briefs/README.md) | What an analyst is handed for a single alerted account |

## What it does

A pipeline over 6,924,049 synthetic transactions from the IBM anti money laundering
dataset. It builds 20 measures for each receiving account. It ranks every account with three
scorers. It cuts the ranking where the day's capacity is spent. It then prices the four outcomes of
that decision. `make all` runs the pipeline end to end in about 3 minutes. It writes
`reports/report.md`, six figures, and one metrics file per stage. 173 tests gate every change. One
test checks for leakage, which means a feature reading a payment from after the moment it scores.
That test fails if the boundary between the feature window and the label window moves by one
instant.

![How raw transactions become an alert budget. Six stages read top to bottom: raw data of 6,924,049 payments is cleaned down to 705,907 accounts over ten usable days, 20 measures per account describe how money moves through it, three scoring methods rank every account and are compared against each other, a drift check flags any feature or score that has moved since training, the ranking is cut at a fixed daily budget, and the final stage sets out what it costs to get that call right or wrong. A single rulebook of shared parameters and definitions feeds every stage, make all runs the whole pipeline in about three minutes, and 173 tests gate every change.](docs/diagrams/pipeline.svg)

## Built with

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.3.3-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4.6-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy_sparse-1.18.0-8CAAE6?logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-337AB7)
![matplotlib](https://img.shields.io/badge/matplotlib-3.11.1-11557C)
![pytest](https://img.shields.io/badge/pytest-9.1.1-0A9EDC?logo=pytest&logoColor=white)
![ruff](https://img.shields.io/badge/ruff-lint-261230)
![mypy](https://img.shields.io/badge/mypy-strict-2A6DB2)

Every version is pinned in `requirements.lock`. The graph features use `scipy.sparse` operations,
not a graph library. The whole feature set stays vectorised, so the pipeline runs on a laptop.

## What we found

The pipeline ranks every account that received money in the last four days and opens the top 400.
It then puts a price on each of the four ways that call can turn out. Three things came out of
that.

**The size of the team decides which accounts get opened.**

Assume a team can work 200 cases a day, so 400 over two days. The pipeline sorts every account by
risk and hands over the top 400. It is a queue with a fixed length. Changing what a mistake costs
changes the price tag on that queue, and the same 400 accounts stay at the front of it. Make a
missed mule cost four times more, or a wrong freeze four times less, and the same 400 come out,
catching the same 5. Double the team and we could open 800 cases and catch 6.

**One account received most of the money.**

The pipeline scored 387 mule accounts in the test window. One of them received 69.3% of all the
money that reached those accounts. The typical case and the average case are nothing alike. The
middle account received 10,230 EUR. The average is 944,448 EUR, ninety-two times higher.

That gap decides whether the work pays for itself. Say a catch saves the typical 10,230 EUR. An
alert then has to be right 0.8730% of the time to cover what it cost to raise. Say a catch saves the
average 944,448 EUR instead. Being right 0.0106% of the time is enough. The queue is right 0.1565%
of the time, which falls between the two. Take the typical figure and the recommendation would be
to stop alerting. Take the average and it would be to alert far more.

**A rule written by hand did as well as XGBoost.**

The rule has two conditions and no model behind it. It flags an account that received an unusual
number of payments and then sent nearly all of that money out again. XGBoost stopped growing after
one tree. Measured on the same accounts, the gap between the two includes zero, so this data
cannot separate them.

Test window, 400 alerts over two days. A random 400 accounts would catch 0.6 of the 387.

| Scorer | Caught of 387 | Money in the accounts it caught |
| --- | --- | --- |
| rules | 1 | 10,257 EUR |
| logistic | 2 | 5,112,631 EUR |
| xgboost | 5 | 58,106 EUR |

XGBoost caught the most accounts. Logistic regression caught two and reached 88 times more money,
because one of its two was a large account. The full comparison is in
[docs/results.md](docs/results.md).

## How to run

```bash
make setup      # venv on Python 3.12, install, write requirements.lock
make all        # ingest -> features -> models -> economics -> monitoring -> report
make hi-small   # the same pipeline on the development variant, into reports/hi-small/
make test       # pytest, including a worked-example test for each of the six formulas
make lint       # ruff check, ruff format --check, mypy --strict
```

The dataset is licence-gated. So `make data` prints the `kaggle` commands and the licence, and
downloads nothing. The source is IBM Transactions for Anti Money Laundering (Altman et al.),
published under the Community Data License Agreement, companion paper arXiv 2306.16424. Put all four
files in `data/raw/`: `LI-Small_Trans.csv`, `LI-Small_Patterns.txt`, `HI-Small_Trans.csv`, and
`HI-Small_Patterns.txt`. Then run `make all`.

## Limits

The data is fully synthetic. So nothing here describes fraud outside the simulator, or how a bank
operates. The simulator planted the laundering patterns by rule. A model that finds them is partly
finding the rule. The amounts are invented and
converted through an invented exchange-rate table, so no euro figure here is money. The economics
rests on four cost numbers. One comes from a published source: 1150 EUR for a missed mule, from
DNB's 2025 payment fraud statistics. The other three are assumptions. They are stated and never
measured.
