# Equal-budget point guidance: frozen dual-protocol experiment

Frozen on 2026-08-13 before outcome inspection.

## Question

Under an equal latent intervention budget, do PCK-ranked Top100 attention heads
provide a more useful direction for moving the generated object toward its
source-video trajectory than Bottom100 or a layer-matched Random100 control?

This is tested independently in two backends. Raw scores are not compared
between backends because model weights, context information, and resolution
differ.

| Backend | Context actually consumed | Observed Query times | Target future Key times | Resolution |
|---|---|---:|---:|---:|
| `firstframe_ti2v` | first RGB frame | `R0` | `R1..R12` | 1280×704 |
| `context8_v2v` | first 8 RGB frames, VAE-compressed to two clean prefix latents | `R0,R1` | `R2..R12` | 896×512 |

The V2V backend uses checkpoint
`full_sa_no_object_xssc_loss_dinov3_movic_step50000/.../step-000500` with its
object branch disabled. It therefore genuinely consumes the 8-frame visual
prefix, but no xSSC object tokens are injected.

## Cohort and matrix

- Source: `test_5.txt`.
- Frozen eligibility: the existing unguided Baseline trajectory gate, selected
  without inspecting any new guided outcome.
- 9 cases, 10 moving-object targets, one seed (`47326`).
- Per backend: 9 Baselines plus 10×3 guided videos = 39 videos.
- Both backends: 78 videos.
- 40 denoising steps are used so the `s039/latest3350` ranking has its original
  step semantics. The shell inference helper's 8-step default is intentionally
  overridden.

## Head controls

- `Top100`: ranks 1–100 of `pck_head_scopes_s039_latest3350.json`.
- `Bottom100`: ranks 621–720.
- `Random100`: frozen `random100_layer_matched_draw0`; its per-layer head count
  exactly matches Top100 and it has no head overlap with Top100 or Bottom100.

For `firstframe_ti2v`, this is the native legacy latest3350 ranking. For
`context8_v2v`, the same heads are transferred into a separately trained
Full-SA checkpoint. Results from that backend therefore test **ranking
transfer**, not checkpoint-native Top100 quality. A checkpoint-native ranking
would require a separate PCK re-ranking experiment.

## Point target and loss

First-frame object points are transported across all 49 source frames by
CoTracker. Frames `F00,F04,...,F48` define 13 point locations. V2V coordinates
are mapped through the exact 896×512 cover-crop geometry; points outside the
crop are marked invisible.

For an observed-context tracked point Query, selected-head logits compete over
every Wan key token in all 13 latent frames:

\[
\log Z(q)=\log\sum_{k\in\Omega_{0:12}}\exp(q^\top k/\sqrt d).
\]

The target is an equal mixture of Gaussians around the same tracked point in
each visible future latent. Thus the CE penalizes wrong spatial correspondence,
wrong future time allocation, and attention that remains in the context; there
is no per-frame re-softmax.

## Direction correction and attention audit: v3

The original `v1` implementation was re-audited on 2026-08-13 and found to use

\[
Q(R_t,p_t^i) \longrightarrow K(R_{ctx},p_{ctx}^i),
\]

where `p_t^i` is already the future CoTracker position. That old reverse-v1
matrix was stopped and retained unchanged under
`wan_context_point_guidance_head_compare/v1`.

The active `attention_audit_v3` experiment retains the corrected forward-v2
loss and uses

\[
Q(R_{ctx},p_{ctx}^i) \longrightarrow K(R_t,p_t^i).
\]

Here the observed point supplies the Query. For each context Query, Wan logits
are normalized once over the complete `13×H×W` Key sequence. The target is an
equal mixture of Gaussians at the same CoTracker point ID over all visible
future latent times. This is equivalent to averaging the per-future-time
cross-entropies while retaining Wan's true global attention denominator.

The new matrix is written separately under
`wan_context_point_guidance_head_compare/attention_audit_v3`; neither v1 nor
forward-v2 RGB results are reused or silently relabelled because their Q/K
activations cannot be reconstructed after generation.

The dashboard therefore displays, for every backend/case/target:

1. the 13-anchor source GT/pseudo-GT CoTracker point trajectory;
2. arrows for the actually implemented context-Query → future-Key constraint;
3. arrows for the archived future-Query → context-Key v1 geometry;
4. the same-backend unguided Baseline output trajectory against source GT.

Item 4 is the **pre-guidance generated output**, not a pre-update attention
argmax trajectory.

### Denoising-step attention microscope

The v3 rerun explicitly captures steps `5,10,15,20,25,30,35,40` using 1-based
denoising numbering. At every captured step the same guided run records:

- `PRE`: global Wan attention at `x_s`, before the normalized latent update;
- `POST`: global Wan attention at `x'_s`, after that update and before the
  scheduler step;
- `POST-PRE`: signed attention change, red for increase and blue for decrease;
- post-guidance predicted clean latent
  `xhat_0 = x'_s - sigma_s * v_CFG(x'_s)` decoded through the frozen VAE.

Each audit video has 13 frames, one per source anchor. Every frame is a
five-panel `Source | PRE | POST | POST-PRE | predicted-x0` comparison. PRE and
POST use the same source-frame background and a shared p99.5 display scale
within that latent time. Display rescaling does not alter the saved raw maps or
metrics. `frame mass` reports the original global-softmax probability assigned
to that time, `localized mass` measures probability inside the same-ID point's
2-sigma neighborhood, `peak distance` measures the token distance to the GT
point, and `hit rate` reports whether the peak falls inside 2 sigma.

A guided output is marked complete only when its final RGB video and all eight
attention-audit step directories are complete.

`launch_dual_gpu3.sh` runs this diagnostic renderer after both generation
backends finish.  Until a same-backend Baseline exists, its trajectory card is
kept as `PENDING`; the three source/geometry audit videos are deterministic and
can be rendered immediately.

## Direct latent update and equal budget

At every guided step:

1. Run the positive conditional branch and compute the global point loss.
2. Compute `g = dL/dx_s`; model parameters remain frozen.
3. Set context-prefix components of `g` to zero.
4. Apply
   \[
   x'_s=x_s-0.01\,g/\operatorname{RMS}_{mutable}(g).
   \]
5. Re-run both positive and negative CFG branches at `x'_s`.
6. Apply the ordinary FlowMatch scheduler step and restore the clean prefix.

Consequently every head group has exactly `RMS_mutable(delta x)=0.01` per
guided step. Each step logs pre/post loss, actual update RMS, context update
maximum, and selected-head count. The run stops if the one-step sanity check
does not reduce loss, alters context, produces non-finite values, or observes
anything other than 100 selected heads.

## Outcomes

Primary outcome within each backend:

- GT Center/Point ADE normalized by first-frame object diameter (`ADE/D0`),
  guided minus the same-backend, same-case, same-seed Baseline. Smaller is
  better; a negative delta is an improvement.

Secondary outcomes:

- FDE/D0, PCK@10%D0, PCK@20%D0.
- Future trajectory quality-gate pass rate and Track Loss, which retain cases
  where a destructive intervention makes the object untrackable.
- Existing survival, identity, appearance, and outside-object metrics are run
  after trajectory generation; an ADE improvement is not accepted as useful
  if disappearance/identity failure materially worsens.

Cases are the highest independent unit. Final means must first average targets
within case and then weight cases equally. This matrix is a discovery test;
any claim of generalization requires a held-out cohort.

## Live visualization decision

The dual-protocol matrix is integrated into the existing 8092 route
`/gt-stc-guidance-results?v=5` instead of creating a disconnected page. The
dashboard reads each backend `task_manifest.json` as the source of planned
work, so it always renders all 78 slots: Baseline, Top100, Bottom100, and
Random100 for each selectable case/target in both protocol rows. A slot changes
from `PENDING` to a lazy-loaded video only after both `generated.mp4` and
`complete.json` exist; metric fields remain independently pending until
`trajectory_metrics.json` is present. The catalog is rescanned every 30
seconds, preserving the selected case/target while generation continues.

The page additionally exposes a global step rail for the eight captured
denoising steps. For each backend it places Top100, Bottom100 and Random100
side-by-side; unavailable step videos remain explicit `PENDING` cards.
