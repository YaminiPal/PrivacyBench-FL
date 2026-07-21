# Feature Showcase

## Adaptive, Personalized, Privacy-Aware Federated Learning

PrivacyBench-FL is designed as more than a FedAvg demo. It is a compact
research-style benchmark that asks a practical question: how do we preserve
utility when clients differ in data distribution, privacy requirements, and
communication capacity?

| Capability | What it does | Why it matters |
|---|---|---|
| Similarity-aware aggregation | Weights updates by cosine agreement with the round consensus | Reduces the influence of highly divergent clients without exposing their data |
| Personalized FL | Shares an MLP encoder but keeps each client classifier head local | Supports client-specific decision boundaries and non-IID data |
| Adaptive quantization | Uses 16-bit updates while learning is volatile and 4-bit updates later | Trades precision for bandwidth only after the model stabilizes |
| Personalized privacy budgets | Tracks an independent RDP epsilon for every client | Models realistic institutions with different privacy policies |
| Drift detection and response | Detects feature-moment shifts and automatically adapts local training | Moves from synthetic drift injection to a responsive system |
| Continual learning | Learns chronologically across T1, T2 and T3, then measures retention | Makes catastrophic forgetting visible and measurable |
| Multi-objective telemetry | Captures utility, privacy, similarity, bandwidth and retention together | Makes trade-offs explainable instead of reporting accuracy alone |

## Resume-ready explanation

> I built a privacy-aware federated-learning benchmark that personalizes the
> model at the client edge, detects distribution shift, respects different
> client privacy budgets, and uses cosine-similarity-aware aggregation to make
> heterogeneous client updates safer and more useful.

This is deliberately understandable engineering: each component is observable
in the experiment log and can be disabled independently for fair ablations.
