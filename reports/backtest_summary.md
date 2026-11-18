| model                           |   smape |   wape |    mae |   bias_pct |   coverage_80 |
|:--------------------------------|--------:|-------:|-------:|-----------:|--------------:|
| Poisson noise floor (reference) |  11.776 | 10.205 |  5.926 |      0     |       nan     |
| Ensemble (LightGBM + profile)   |  11.837 | 10.279 |  5.958 |      0.224 |       nan     |
| Multiplicative profile          |  11.855 | 10.304 |  5.977 |      0.821 |       nan     |
| LightGBM (global, direct)       |  11.925 | 10.375 |  6.013 |     -0.374 |         0.789 |
| Seasonal naive (last year)      |  17.15  | 14.727 |  8.538 |     -3.633 |       nan     |
| Seasonal naive (last week)      |  21.695 | 19.61  | 11.443 |     -2.953 |       nan     |