# Phase 12 Final Validation Report

## Overall Strategy Performance (Averages per GW)
|   horizon | strategy      |   net_points |   gross_points |   hits |   bench_points |   transfers_made |
|----------:|:--------------|-------------:|---------------:|-------:|---------------:|-----------------:|
|         1 | Attack        |         39.5 |           55.5 |    4   |           10.5 |              5   |
|         1 | Defend        |         35.5 |           45.5 |    2.5 |            9.5 |              3.5 |
|         1 | Neutral       |         35.5 |           45.5 |    2.5 |            9.5 |              3.5 |
|         1 | Pure xP       |         35.5 |           45.5 |    2.5 |            9.5 |              3.5 |
|         1 | Risk-adjusted |         35.5 |           45.5 |    2.5 |            9.5 |              3.5 |
|         3 | Attack        |         34.5 |           46.5 |    3   |           10.5 |              4   |
|         3 | Defend        |         40   |           52   |    3   |            9.5 |              4   |
|         3 | Neutral       |         34   |           46   |    3   |           10   |              4   |
|         3 | Pure xP       |         33   |           43   |    2.5 |           11   |              3.5 |
|         3 | Risk-adjusted |         44   |           54   |    2.5 |            9.5 |              3.5 |
|         5 | Attack        |         15   |           37   |    5.5 |            9.5 |              6.5 |
|         5 | Defend        |         32.5 |           42.5 |    2.5 |            9.5 |              3.5 |
|         5 | Neutral       |         35   |           57   |    5.5 |            8.5 |              6.5 |
|         5 | Pure xP       |         14   |           40   |    6.5 |            7.5 |              7.5 |
|         5 | Risk-adjusted |         27.5 |           49.5 |    5.5 |            8   |              6.5 |
|         8 | Attack        |          2   |           30   |    7   |            7.5 |              7.5 |
|         8 | Defend        |          3.5 |           31.5 |    7   |            7   |              7.5 |
|         8 | Neutral       |         10   |           40   |    7.5 |            7   |              8   |
|         8 | Pure xP       |          1.5 |           31.5 |    7.5 |            7   |              8.5 |
|         8 | Risk-adjusted |         10.5 |           40.5 |    7.5 |           10.5 |              8.5 |

## Solver Reliability
|   horizon | strategy      |   FEASIBLE |   OPTIMAL |
|----------:|:--------------|-----------:|----------:|
|         1 | Attack        |          0 |         2 |
|         1 | Defend        |          0 |         2 |
|         1 | Neutral       |          0 |         2 |
|         1 | Pure xP       |          0 |         2 |
|         1 | Risk-adjusted |          0 |         2 |
|         3 | Attack        |          2 |         0 |
|         3 | Defend        |          1 |         1 |
|         3 | Neutral       |          2 |         0 |
|         3 | Pure xP       |          2 |         0 |
|         3 | Risk-adjusted |          2 |         0 |
|         5 | Attack        |          2 |         0 |
|         5 | Defend        |          2 |         0 |
|         5 | Neutral       |          2 |         0 |
|         5 | Pure xP       |          2 |         0 |
|         5 | Risk-adjusted |          2 |         0 |
|         8 | Attack        |          2 |         0 |
|         8 | Defend        |          2 |         0 |
|         8 | Neutral       |          2 |         0 |
|         8 | Pure xP       |          2 |         0 |
|         8 | Risk-adjusted |          2 |         0 |

## Legality Check
|   horizon | legal_squad   |
|----------:|:--------------|
|         1 | 100.0%        |
|         3 | 100.0%        |
|         5 | 90.0%         |
|         8 | 60.0%         |

## Runtime Analysis (Seconds)
|   horizon |   runtime |
|----------:|----------:|
|         1 |      1.11 |
|         3 |     30.65 |
|         5 |     31.52 |
|         8 |     31.79 |

