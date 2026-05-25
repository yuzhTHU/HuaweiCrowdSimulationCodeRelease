# Supplementary Experiment Results for Unified Training

> This section demonstrates that by performing unified training using only GC and SDD, we can prevent WayMo's massive data volume from dominating the training process, which previously caused the model to be unfit on GC and SDD and led to a performance drop for both. This is prepared as supplementary material in response to reviewer cust's second concern during the KDD Rebuttal.

In Table 2, although the model trained jointly on GC, SDD, and WayMo achieved better performance on WayMo compared to the model trained solely on WayMo, its performance on GC and SDD actually decreased compared to the separately trained models. This primarily stems from the massive disparity in sample sizes among the three datasets, which caused WayMo to dominate the training process and led the model to be unfit on GC and SDD. To illustrate this point, we conducted additional experiments following the experimental setup in Table 2, but using only the GC and SDD datasets. The experimental results are shown below. Compared to the models trained independently on GC and SDD, the unified trained model achieved better performance. This demonstrates the effectiveness of the unified training strategy.

| Model | GC | SDD |
| --- | --- | --- |
| Ours (Seperated) | 0.1891 / 0.3537 / 3.24 | 0.3057 / 0.5863 / 1.02 |
| Ours (Unified, GC & SDD) | 0.1844 / 0.3540 / 1.48 | 0.3017 / 0.5803 / 0.37 |