# Depth-as-time on ELF-B-PD: analysis plan (v2)

Status: plan only. No model training. Frozen models, closed-form probes. 1 test GPU per stage.

## Why v1 is not conclusive

The v1 probes (`probe_elf.py`, `probe_handoff*.py`) followed the FMLM recipe too closely:

* **Readout.** We read each block with `final_layer(h_i)`, a head trained only on h_12. The
  intermediate readouts are off-manifold: norm 2-3x the output, cosine to the output 0.15-0.3.
  A negative result with this lens cannot tell "no time-like state" from "the lens cannot read it".
* **Token view.** Token readouts pass the off-manifold latent through a decode pass trained on
  clean-ish latents, so the errors compound.
* **Ruler.** The ruler was an ODE on a uniform 64-step grid with self-conditioning chained. That is
  not the student's distillation target (see below). Student/ruler token agreement was 0.22.
* **Distance.** Latent cosine or L2 against z_t is dominated by the noise term (noise std 2 vs ~1
  for the data), so tau_cos mostly measured the noise fraction.
* **Handoff.** Self-conditioning was reset to 0 at the handoff. Even the teacher's own state then
  reproduced only 50-80% of its tokens, so the test was noisy.

## ELF-specific facts that constrain the analysis

Checked in the vendored code (branch `distillation`, commit 346f00b).

| ELF design | What it implies | How v2 handles it |
|---|---|---|
| **PD round 5 trains only (t_s, t_t) = (0, 1).** Target = endpoint of 64 teacher Euler sub-steps on a *random logit-normal grid* (sorted sigmoid(N(-1.5, 0.8)) points). SC-CFG w ~ log-U[0.5, 5] goes to teacher and student. Self-cond: with p=0.5 the first sub-step and the student get the teacher's uncond x_pred(z0, 0), else zeros; the teacher chains its own x_pred afterwards. | The student is not a flow map. There is no self ruler, and no (s,t) query to read a block "at a time". The right reference for a 1-step sample (x_prev = 0, w = 2.5) is the PD target procedure with use_sc = 0. | A1 rebuilds the ruler exactly as `teacher_pd_target` (deterministic logit-normal quantile grid, plus a few random grids for spread). No student multi-step ruler. |
| **Self-conditioning** input [z, x_prev] -> `self_cond_proj`. | Every state handed to the teacher needs an x_prev. Resetting it changes the trajectory (v1: 50-80% self-agreement). | A4 measures the reset effect on the teacher's own path. Handoffs use two conventions, x_prev = 0 and x_prev = probe-predicted x_hat, each against an own-state baseline with the same convention. |
| **SC-CFG** is a conditioning token (scale w), not a cond/uncond mix at inference. | The student imitates the *guided* teacher at the same w. | The ruler uses the same w = 2.5 token. Sensitivity check at w in {1, 2.5, 4}. |
| **Time enters only through 4 in-context prefix tokens.** No AdaLN; `final_layer` is time-agnostic. | There is no time knob on the readout (unlike FMLM's W_out(h; tau, tau')). | Time is read by probing against a time-indexed target, not by re-conditioning the head. |
| **Bottleneck input projection** 512 -> 128 -> 768. | The network sees only a 128-d projection of z0 per token. The noise part of z_t is at most rank-128 decodable. | The primary probe targets are clean estimates (x_hat_t, delta_{0,t}). z_t is a secondary target, read against the h_0 control. |
| **Latent**: frozen T5-small encoder output, contextual, /0.2. Noise scale 2. | Tokens are not tokenwise one-hot. A latent at one position carries context. Per-dim SNR = t / (2(1 - t)), equal at t = 2/3. | Metrics are R^2 or cosine in latent space, and token agreement only through a calibrated viewer (below). No KL or simplex metrics. |
| **Training time distribution** logit-normal(-1.5, 0.8): median t = 0.18, 95% in [0.04, 0.54]. | ELF does its work at small t. A uniform-t axis wastes resolution. FLM's Pe-based tau does not apply. | A3 measures an **ELF progress clock**: the fraction of tokens already equal to the final ones along the ruler. Results are reported in both t and progress. |
| **Decoder pass**: same 12 blocks, mode tokens on, t = 1, x_prev = 0, w sampled. Trained on per-token noisy latents lambda*x + (1 - lambda)*5*eps, logit(lambda) ~ N(0.8, 0.8). | The decoder is a usable **viewer for noisy latents**, but only inside its training range. For z_t the matching input is (lambda/t)*z_t with lambda/(1 - lambda) = 2.5 t/(1 - t). That is in range for t >~ 0.15. | All token views go through this calibrated viewer, and states below t = 0.15 are flagged. The decode pass's own depth profile is out of scope. |
| **x-prediction**, v = (x_hat - z)/max(1 - t, 0.05). | The clamp distorts v for t > 0.95. | The ruler grid follows the PD sub-step grid; no special handling near 1. |

## Stages

Each stage has a gate. Later stages run only if the gate passes.

### A. Get the reference right
* **A1 ruler.** PD-faithful teacher trajectory from the same z0: 64 Euler sub-steps on the
  logit-normal grid, w = 2.5, x_prev = 0 at the first sub-step then chained. Record z_t, x_hat_t
  and delta_{0,t} = (z_t - (1 - t) z0)/t at every sub-step.
* **A2 does the student track its target?** Compare the student output x_hat with the ruler endpoint
  z_1: latent cosine, relative error, and token agreement through the decoder. This replaces the
  v1 0.22, which used a different ruler. Also check spread across random grids.
  * Gate: if token agreement < ~0.5, the student does not reproduce the teacher path pointwise.
    Then "block i = teacher state at t_i" is ill-posed, and B is reported against both the
    teacher path and the student's own output only.
* **A3 progress clock and SNR.** Along the ruler, measure the viewer-token agreement of z_t and of
  x_hat_t with the final tokens, and the signal and noise norms. This defines where ELF decides.
* **A4 self-cond sensitivity.** Restart the teacher from its own z_t with x_prev = 0 vs the chained
  x_prev. Measure final-token agreement as a function of t. This fixes the handoff convention and
  the noise floor for C.

### B. Read the blocks without the native lens
* **B1 ridge probes** (closed form, frozen models). For every block i and ruler time t, fit
  h_i -> target_t on 48 sequences (49k tokens) and evaluate held-out R^2 on 16 other sequences.
  The split is by sequence. lambda is chosen on a validation split. Targets: x_hat_t (primary),
  delta_{0,t}, and z_t (secondary).
  * Output: an R^2(block, t) heatmap and t*_i = argmax_t.
  * Controls on the same tokens: (i) h_0; (ii) every block of a randomly initialized ELF-B fed the
    same input; (iii) shuffled targets across sequences; (iv) the student's final output x_hat as
    the target, to separate "contains the answer" from "contains a time-t state".
  * Gate (depth-as-time evidence): t*_i increases with i and the peak is interior, so that
    R^2(i, t*_i) clearly exceeds both R^2(i, t -> 0) and R^2(i, answer). Controls (i)-(iii) must
    not show the same pattern.
* **B2 native lens** (`final_layer`), kept only as a secondary row for comparison with v1.
* **B3 views.** Probe-predicted x_hat_{t*_i} for each block, passed through the calibrated viewer,
  next to the ruler's own view at t*_i. This gives text panels per block.

### C. Functional test: is the block state usable as the teacher's state?
Run only if B passes. Hand the probe-predicted state z_{t*_i} (with x_prev by both A4
conventions) to the teacher at t*_i and finish on the PD grid. Compare with these baselines:
* own: the teacher's z_{t*_i}, same x_prev convention
* answer: (1 - t) z0 + t x_hat_student, placed at the same t
* noise: z0 only

The C metrics are final-token agreement with the student and with the teacher, Gen. PPL, and
entropy.

### D. Per-token structure
* The dependency test (resample z0 at position j, measure output change at i) does not need a lens.
  It can run regardless of B.
* Commit order and confidence ordering only if B passes, with commit defined on viewer tokens of
  the probed states.

## Not ported from FMLM, and why
* The tau reparameterization (Pe LUT) and the noise-seeding formula softmax(log m + beta z). Both
  come from the one-hot Gaussian posterior and do not hold in the T5 latent.
* KL or simplex distances. The latent is Euclidean.
* The self ruler. The PD student is not a flow map.
* The (1 - t) z + t delta handoff built from the native lens. It is replaced by probe-predicted
  states.
* Layer reuse and compression. Out of scope until B passes. The decoder shares the blocks.

## FMLM cross-check
Apply B1 to FMLM with the same protocol so both models are read with the same instrument. The
target is the teacher's final hidden state on (x_t, t), 768-d in both models. That avoids the
50k-dim simplex.

## Cost
A: about 15 min. B1: about 20 min (one X^T X per block, all targets solved together). C: about
30 min. D-dependency: about 20 min. All on `beta_test`, 1 GPU.

## Results (2026-10-25)

Files: `results/elf/v2/stageA.json`, `stageA2b.json`, `stageB_centered.json`. The uncentred
`stageB.json` is confounded; see below.

**A1.** The PD-faithful ruler is a good sampler: Gen. PPL 29.0 / entropy 5.19 on 64 samples. The v1
uniform-grid ruler gives 69.5 / 5.43.

**A2 gate fails.** Student vs ruler token agreement is 0.18; chance is ~0.02. The teacher itself is
grid-sensitive: two random PD grids agree on 0.42-0.46 of tokens (latent cosine 0.62-0.65).

**A2b.** The student is closer to the grid-averaged teacher endpoint (cosine 0.67) than to a
single-grid endpoint (0.55). For reference, a single grid vs the leave-one-out mean of 7 others is
0.76. Its norm is shrunk: student 359, 8-grid mean 504, single grid 617. This is consistent with MSE
PD regressing onto E_grid[z_1 | z0].

**A3.** The decision window is t ~ 0.12-0.30. Over it, decode(x_hat_t) agreement with the final
tokens rises 0.15 -> 0.89. The calibrated viewer of z_t lags, at 0.07 -> 0.62.

**A4.** Resetting self-cond at t < 0.2 changes about half of the endpoint tokens (agreement
0.37-0.55). A fresh uncond seed is only slightly better than zeros. Recovery to 0.82-0.93 needs
t >= 0.28.

**B, centred.** The uncentred run was confounded by a per-position mean: shuffled targets gave
R^2 0.76. After centring, the shuffled control is ~0 and the random-init network ~0.02.

| | b0 | b2 | b4 | b6 | b8 | b10 | b12 |
|---|---|---|---|---|---|---|---|
| t* (x_hat_t) | .04 | .04 | .07 | .09 | .09 | .10 | .10 |
| R^2 at t* | .11 | .85 | .84 | .77 | .69 | .65 | .77 |
| R^2 for t = 1 (teacher endpoint) | .02 | .11 | .13 | .15 | .15 | .16 | .20 |
| R^2 for the student's own output | .08 | .47 | .59 | .66 | .72 | .78 | .99 |

The delta_{0,t} target gives t* from .06 to .13.

* Blocks hold the teacher's clean estimate x_hat_t linearly, and the best-matching t drifts upward
  with depth. Early blocks match the teacher's estimate better than the student's own answer
  (b2: .85 vs .47).
* The whole drift lies at t <= 0.13, before the decision window. Even the final block matches the
  teacher at t ~ 0.1 (R^2 .77), not its endpoint (.20).
* Reading: the one-step PD student outputs roughly the teacher's early posterior mean, and the
  decoder turns that blurry latent into tokens. Depth spans only the pre-decision part of the flow.
  FMLM's 12 blocks span its decision window, t 0.71-0.87.
* C (handoff) is not informative at t <= 0.13. The A4 floor there is ~0.5, so it was not run.
