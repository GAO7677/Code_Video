# Cross-model common attention heads for case 001460

## Scope

This document records self-attention heads shared by Wan+LoRA, Wan+xSSC, and
PhysRVG for `0613pybullet_sample_001460_w002`. The query is the exact moving
ball region at output frame 8, represented by four latent patches:
`(2,6,13)`, `(2,6,14)`, `(2,7,13)`, and `(2,7,14)`. Attention was captured at
denoising steps 5, 15, 25, and 35.

A head is called common when the same Block/Head pair has the same primary
role in all three models, is classified as clear in every model, and keeps
that role for at least three of the four denoising steps in every model.

## Recorded heads

| Role | Block/Head | Assignment | Evidence |
|---|---|---|---|
| T | B28-H19 | Moving-ball trajectory propagation | Strongest common trajectory head; stable 4/4, 4/4, 3/4 |
| T | B19-H12 | Moving-ball trajectory propagation | Most temporally stable trajectory head; stable 4/4 in all models |
| T | B6-H2 | Trajectory with fixed-position alignment | Stable 4/4 in all models; retains a secondary P tendency |
| T | B2-H14 | Early trajectory/temporal correspondence | Same role in all models, but stable 3/4 in each |
| S | B27-H2 | Intraframe spatial attention | Clear S in all models |
| S | B25-H12 | Intraframe spatial attention | Clear S in all models |
| S | B29-H7 | Intraframe spatial attention | Clear S in all models |
| S | B28-H22 | Intraframe spatial attention | Clear S in all models |
| P | B12-H18 | Fixed-position temporal alignment | Strongest common P head |
| P | B28-H12 | Fixed-position temporal alignment | Clear P in all models |
| P | B8-H8 | Fixed-position temporal alignment | Clear P in all models |
| C | B3-H8 | First-frame/history context | Strongest common C head |
| C | B22-H15 | First-frame/history context | Clear C in all models |
| C | B0-H5 | First-frame/history context | Clear C in all models |
| G | B10-H7 | Global aggregation | Strongest common G head |
| G | B9-H21 | Global aggregation | Clear G in all models |
| G | B7-H18 | Global aggregation | Clear G in all models |

## Strong common trajectory evidence

| Block/Head | Wan+LoRA consistency | Wan+xSSC consistency | PhysRVG consistency | Cross-ball enrichment: LoRA / xSSC / PhysRVG |
|---|---:|---:|---:|---:|
| B28-H19 | 4/4 | 4/4 | 3/4 | 26.64 / 44.83 / 44.97 |
| B19-H12 | 4/4 | 4/4 | 4/4 | 15.36 / 22.86 / 26.16 |
| B6-H2 | 4/4 | 4/4 | 4/4 | 18.08 / 50.73 / 41.09 |
| B2-H14 | 3/4 | 3/4 | 3/4 | 11.39 / 50.66 / 45.71 |

## Interpretation boundary

These are strong cross-configuration observations for one video and one
moving-object query, not universal or causal head semantics. All three models
share the Wan backbone and much of its pretrained attention structure.
Therefore, common heads likely represent functions inherited from Wan and
preserved after LoRA, xSSC, or PhysRVG adaptation. Validation on more videos,
objects, trajectories, and query times is required before making a general
claim.

