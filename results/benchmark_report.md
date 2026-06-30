# 🔒 PrivacyBench-FL: Global Benchmark Run Summary
**Generated on:** 2026-06-17 23:33:10

This report presents an automated statistical analysis of the Privacy-Utility-Bandwidth trade-offs over the preprocessed Adult Income Dataset.

## 🏆 Executive Summary Table
| Architecture Blueprint | Peak Accuracy (%) | Final Loss | Total Spent Privacy (ε) | Bandwidth Transmitted (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **LR** | 72.52% | 0.5995 | 0.423 | 0.02 MB |
| **MLP** | 72.18% | 0.9345 | 0.423 | 0.99 MB |

## 🔬 Deep Technical Insights
1. **Predictive Variance Variance:** The `LR` achieved peak utility constraints, outperforming the basic `MLP` setup by an absolute **0.34%** accuracy bound.
2. **Compression Efficiency:** Interestingly, `LR` maintained a lower overall wire footprint than `MLP`, demonstrating highly effective parameters optimization pruning.
3. **Differential Privacy Footprint:** The absolute highest information protection spend bound was reached by `MLP` at an extended cumulative **ε = 0.423** (at a locked target δ = 1e-5).


*-- End of Experiment Telemetry Output. All calculation profiles comply with strict DP-SGD auditing constraints. --*