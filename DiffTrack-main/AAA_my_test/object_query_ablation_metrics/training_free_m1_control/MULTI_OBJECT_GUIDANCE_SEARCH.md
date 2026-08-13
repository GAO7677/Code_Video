# Multi-object M1 Guidance Search

## Frozen intervention

One generated video jointly perturbs every detected object.  For frozen
same-seed Baseline tubes `R_i`, the perturbed conditional forward is

\[
Y'_{R_i,h}=Y_{R_i,h}-A_h[R_i,R_i]V_h[R_i],
\qquad h\in\text{latest3350 Top100}.
\]

The implementation deletes the set union
`union_i (R_i x R_i)`.  It does **not** form one union object and therefore
preserves `A[R_i,R_j]V[R_j]` for `i != j`.  When two objects occupy one latent
cell, token pairs are set-unioned so the same contribution is never subtracted
twice; every output manifest records the overlap and deduplication count.

Guidance uses

\[
\epsilon=\epsilon_u+5(\epsilon_c-\epsilon_u)
+\lambda(\epsilon_c-\epsilon_{c,M1\text{-multi}}).
\]

Only the conditional branch receives the contrast term.  The object tubes are
tracked on the same-seed no-intervention Baseline.  Future source-video/GT
trajectories are not used during guidance.

## Frozen matrix

- Input: `/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt`
- 20 cases.
- Seeds: `13248, 32466, 47326, 68613, 90094`; seed `35075` excluded.
- `lambda`: `-1, -0.5, +0.5, +1`; `lambda=0` reuses Baseline.
- Inclusive windows: `0-4, 0-9, 0-19, 0-39`.
- 100 Baselines + 1,600 guided videos = 1,700 videos.
- GPUs: physical GPU 1 and GPU 2, 50 case-seed units / 800 guided tasks each.

## Runtime files

- Manifest: `/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_m1_multi_object_search_v1/search_manifest.json`
- Outputs: `/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_m1_multi_object_search_v1`
- Runner: `run_multi_object_guidance_search.py`
- Launcher: `launch_multi_object_guidance_search_gpu12.sh`
- Test: `test_multi_object_guidance_search.py`
- tmux: `m1_multi_object_search_gpu12:gpu1` and `:gpu2`.

## Post-generation selection

MSE and CoTracker trajectory metrics are offline selection criteria only; they
do not enter the guidance computation.  Hyperparameters must be compared on
paired case×seed units, then averaged within case before case-balanced summary.
Trackability/retention failures must remain in the report rather than being
silently removed by the trajectory quality gate.
