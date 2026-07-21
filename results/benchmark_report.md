# 🔒 PrivacyBench-FL: Global Benchmark Run Summary
**Generated on:** 2026-07-22 00:03:26

This report presents an automated statistical analysis of the Privacy-Utility-Bandwidth trade-offs over the preprocessed Adult Income Dataset.

## 🏆 Executive Summary Table
| Architecture Blueprint | Peak Accuracy (%) | Final Loss | Total Spent Privacy (ε) | Bandwidth Transmitted (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **LR** | 71.66% | 0.5944 | 2.179 | 0.04 MB |
| **MLP** | 70.83% | 1.0387 | 1.612 | 0.24 MB |

## 🔬 Deep Technical Insights
1. **Predictive Variance Variance:** The `LR` achieved peak utility constraints, outperforming the basic `MLP` setup by an absolute **0.83%** accuracy bound.
2. **Compression Efficiency:** Interestingly, `LR` maintained a lower overall wire footprint than `MLP`, demonstrating highly effective parameters optimization pruning.
3. **Differential Privacy Footprint:** The absolute highest information protection spend bound was reached by `LR` at an extended cumulative **ε = 2.179** (at a locked target δ = 1e-5).


*-- End of Experiment Telemetry Output. All calculation profiles comply with strict DP-SGD auditing constraints. --*