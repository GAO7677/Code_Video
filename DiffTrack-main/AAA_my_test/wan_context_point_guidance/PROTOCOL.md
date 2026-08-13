# Equal-budget point guidance: frozen dual-protocol experiment

Frozen on 2026-08-13 before outcome inspection.

## Question

Under an equal latent intervention budget, do PCK-ranked Top100 attention heads
provide a more useful direction for moving the generated object toward its
source-video trajectory than Bottom100 or a layer-matched Random100 control?

This is tested independently in two backends. Raw scores are not compared
between backends because model weights, context information, and resolution
differ.

| Backend | Context actually consumed | Guided Query times | Target Key times | Resolution |
|---|---|---:|---:|---:|
| `firstframe_ti2v` | first RGB frame | `R1..R12` | `R0` | 1280×704 |
| `context8_v2v` | first 8 RGB frames, VAE-compressed to two clean prefix latents | `R2..R12` | `R0,R1` | 896×512 |

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

For a future tracked point Query, selected-head logits compete over every Wan
key token in all 13 latent frames:

\[
\log Z(q)=\log\sum_{k\in\Omega_{0:12}}\exp(q^\top k/\sqrt d).
\]

The target is a Gaussian around the same tracked point in each visible context
latent. V2V uses an equal mixture over `R0` and `R1`. Thus the CE penalizes both
wrong spatial correspondence and failure to read the intended context time;
there is no per-frame re-softmax.

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

