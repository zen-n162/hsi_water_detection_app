# Baseline Exact Area-cap Ranking Summary

Exact area-cap ranking selects exactly the requested number of sampled valid pixels by score.
Ties are broken by a deterministic random key, which is useful for patch-scalar models whose scores are repeated over many pixels.

| policy | pos rate mean | F1 mean | random F1 | F1-random |
|---|---:|---:|---:|---:|
| area_cap_15_exact_rank | 0.1500 | 0.1178 | 0.0964 | 0.0214 |
| area_cap_20_exact_rank | 0.2000 | 0.1199 | 0.1047 | 0.0152 |
