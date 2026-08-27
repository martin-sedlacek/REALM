# Reproducibility

The results in the paper were produced with REALM `v0.1.1`. The benchmark and simulator have
changed since then, so use that release when reproducing the published numbers:

```sh
git fetch --tags
git checkout v0.1.1
```

Run every combination of the 10 tasks and 16 perturbation settings with:

- 25 repeats per task and perturbation;
- 800 time-steps per rollout.

This is 4,000 rollouts per model. Each table entry is the mean over the 10 tasks for that
perturbation.

The GR00T N1.5 result uses our custom fine-tuned checkpoint. It is not the result of running the
public base GR00T N1.5 model without fine-tuning.

## Paper results

| Perturbation | **$\pi_0$** | **$\pi_0$-FAST** | **GR00T N1.5** |
|:---|:---:|:---:|:---:|
| **Default** | 0.44 | 0.61 | 0.19 |
| **V-AUG** | 0.42 (-0.02 $\downarrow$) | 0.64 (+0.03 $\uparrow$) | 0.19 (-0.00 -) |
| **V-VIEW** | 0.52 (+0.08 $\uparrow$) | 0.70 (+0.09 $\uparrow$) | 0.19 (-0.00 -) |
| **V-SC** | 0.43 (-0.01 $\downarrow$) | 0.60 (-0.02 $\downarrow$) | 0.21 (+0.02 $\uparrow$) |
| **V-LIGHT** | 0.37 (-0.07 $\downarrow$) | 0.54 (-0.07 $\downarrow$) | 0.16 (-0.03 $\downarrow$) |
| **S-PROP** | 0.29 (-0.15 $\downarrow$) | 0.53 (-0.08 $\downarrow$) | 0.21 (+0.02 $\uparrow$) |
| **S-LANG** | 0.36 (-0.08 $\downarrow$) | 0.61 (-0.01 $\downarrow$) | 0.21 (+0.02 $\uparrow$) |
| **S-MO** | 0.35 (-0.09 $\downarrow$) | 0.55 (-0.06 $\downarrow$) | 0.20 (+0.01 $\uparrow$) |
| **S-AFF** | 0.30 (-0.14 $\downarrow$) | 0.55 (-0.06 $\downarrow$) | 0.21 (+0.01 $\uparrow$) |
| **S-INT** | 0.29 (-0.15 $\downarrow$) | 0.54 (-0.07 $\downarrow$) | 0.20 (+0.01 $\uparrow$) |
| **B-HOBJ** | 0.32 (-0.12 $\downarrow$) | 0.38 (-0.23 $\downarrow$) | 0.16 (-0.03 $\downarrow$) |
| **SB-NOUN** | 0.28 (-0.16 $\downarrow$) | 0.39 (-0.22 $\downarrow$) | 0.17 (-0.02 $\downarrow$) |
| **SB-VRB** | 0.36 (-0.08 $\downarrow$) | 0.57 (-0.04 $\downarrow$) | 0.21 (+0.02 $\uparrow$) |
| **VB-POSE** | 0.32 (-0.12 $\downarrow$) | 0.49 (-0.12 $\downarrow$) | 0.07 (-0.12 $\downarrow$) |
| **VB-MOBJ** | 0.38 (-0.06 $\downarrow$) | 0.53 (-0.09 $\downarrow$) | 0.09 (-0.10 $\downarrow$) |
| **VSB-NOBJ** | 0.16 (-0.28 $\downarrow$) | 0.26 (-0.35 $\downarrow$) | 0.09 (-0.10 $\downarrow$) |
| **V-Avg.** | 0.37 (-0.07 $\downarrow$) | 0.54 (-0.08 $\downarrow$) | 0.14 (-0.05 $\downarrow$) |
| **S-Avg.** | 0.30 (-0.14 $\downarrow$) | 0.50 (-0.11 $\downarrow$) | 0.19 (-0.00 -) |
| **B-Avg.** | 0.30 (-0.13 $\downarrow$) | 0.44 (-0.17 $\downarrow$) | 0.13 (-0.06 $\downarrow$) |
