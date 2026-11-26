# Matched proposal–zero differences in Table 6

The four branches of each F1 scenario share the same physical problem and zero-proposal baseline. Each historical model contributes 60 scenarios and 240 matched branch differences. The printed means, medians and three-way sign counts reproduce directly.

The caption identifies the interval as a scenario-resampling stability interval. Sampling the 240 branch differences independently with replacement reproduces the eight printed endpoints at their displayed precision. Sampling the 60 scenario means with replacement keeps the four branches together. The script runs both calculations with 10,000 draws, seed 20260716 and percentile limits.

| Model | Contrast | Published interval and reproduced branch interval | Scenario-cluster interval |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash | Correction, kWh | [187.74, 510.46] | [93.45, 643.30] |
| DeepSeek-V4-Flash | Regret, USD | [−0.330, 1.006] | [−0.493, 1.351] |
| Qwen3.6-Flash | Correction, kWh | [4489.48, 4774.67] | [4361.71, 4902.94] |
| Qwen3.6-Flash | Regret, USD | [−0.245, −0.035] | [−0.261, −0.023] |

The direction of each interval relative to zero agrees across the two calculations. The numerical agreement under branch resampling leaves the historical code invocation unestablished.
