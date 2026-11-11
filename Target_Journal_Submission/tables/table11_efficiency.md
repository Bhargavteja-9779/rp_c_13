| component                                                       |      m |       ms |   peak_mb |
|:----------------------------------------------------------------|-------:|---------:|----------:|
| fit decision layer (once, incl. 50 bootstrap static thresholds) |   9766 | 3696.234 |    14.192 |
| static threshold (comparison only)                              |    100 |    0.001 |   nan     |
| EM prior correction (SLD)                                       |    100 |    0.149 |   nan     |
| DTA-EM (EM + sort + expected-F scan)                            |    100 |    0.166 |   nan     |
| PAFT-Plug                                                       |    100 |    0.036 |   nan     |
| PAFT-Bayes                                                      |    100 |    0.429 |   nan     |
| PAFT guard (KS test)                                            |    100 |    0.066 |   nan     |
| all rules (decide)                                              |    100 |    1.149 |   nan     |
| static threshold (comparison only)                              |    500 |    0.003 |   nan     |
| EM prior correction (SLD)                                       |    500 |    0.186 |   nan     |
| DTA-EM (EM + sort + expected-F scan)                            |    500 |    0.184 |   nan     |
| PAFT-Plug                                                       |    500 |    0.042 |   nan     |
| PAFT-Bayes                                                      |    500 |    0.498 |   nan     |
| PAFT guard (KS test)                                            |    500 |    0.089 |   nan     |
| all rules (decide)                                              |    500 |    1.283 |   nan     |
| static threshold (comparison only)                              |   2000 |    0.001 |   nan     |
| EM prior correction (SLD)                                       |   2000 |    0.270 |   nan     |
| DTA-EM (EM + sort + expected-F scan)                            |   2000 |    0.281 |   nan     |
| PAFT-Plug                                                       |   2000 |    0.081 |   nan     |
| PAFT-Bayes                                                      |   2000 |    0.558 |   nan     |
| PAFT guard (KS test)                                            |   2000 |    0.194 |   nan     |
| all rules (decide)                                              |   2000 |    1.513 |   nan     |
| static threshold (comparison only)                              |  10000 |    0.001 |   nan     |
| EM prior correction (SLD)                                       |  10000 |    0.727 |   nan     |
| DTA-EM (EM + sort + expected-F scan)                            |  10000 |    0.886 |   nan     |
| PAFT-Plug                                                       |  10000 |    0.322 |   nan     |
| PAFT-Bayes                                                      |  10000 |    0.765 |   nan     |
| PAFT guard (KS test)                                            |  10000 |    0.567 |   nan     |
| all rules (decide)                                              |  10000 |    3.040 |   nan     |
| static threshold (comparison only)                              | 100000 |    0.009 |   nan     |
| EM prior correction (SLD)                                       | 100000 |   13.237 |   nan     |
| DTA-EM (EM + sort + expected-F scan)                            | 100000 |   10.183 |   nan     |
| PAFT-Plug                                                       | 100000 |    2.852 |   nan     |
| PAFT-Bayes                                                      | 100000 |    3.162 |   nan     |
| PAFT guard (KS test)                                            | 100000 |    4.333 |   nan     |
| all rules (decide)                                              | 100000 |   29.978 |   nan     |
| retrain LightGBM on adult (reference)                           |  24424 | 1511.209 |   nan     |
| stored state: thresholds T x grid G surface (float64)           | 329728 |  nan     |     2.516 |