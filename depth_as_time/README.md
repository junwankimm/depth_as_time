# Depth as Time in a one-step *language* flow map (FMLM)

Question: does the "depth as time" observation of
[Asiimwe et al., arXiv:2610.03626](../papers/2610.03626.pdf) — one-step generators
unroll the multi-step denoising trajectory across network depth, readable with the
model's own output head — also hold for language, using the one-step Flow Map
Language Model (FMLM) of [Lee et al., arXiv:2602.16813](../papers/2602.16813.pdf)?

Models: released LM1B checkpoints `david3684/FLM-B-LM1B` (teacher flow, 12-block
DiT, |V| = 30,522, L = 128) and `david3684/FMLM-B-LM1B` (flow map distilled from it).
Sanity: the depth-12 readout equals the model output exactly (max abs diff 0);
one-step FMLM Gen. PPL 118.1 / entropy 4.14 (paper: 119.3 / 4.16); FLM 128-step
117.4 / 4.34 (paper at 128 steps: 112.5 / 4.34), each on 256 samples.

## 1. How to look at a "noisy intermediate state" in token space

FLM is a *continuous* flow on one-hot vectors, x_t = (1 - t) z + t onehot(y),
z ~ N(0, I) in R^{L x |V|} (t = 0 noise, t = 1 data). So the image machinery
transfers, but "noise" has two different faces in token space:

| | vision | language (FLM) |
|---|---|---|
| state x_t | noisy image | argmax(x_t) = true token w.p. 1 - P_e(t), otherwise a random vocab item. FLM's time reparameterisation tau(t) is *exactly* "fraction of tokens decided" (Eq. 25). |
| clean prediction E[x_1 \| x_t] | blurry image, mean image at full noise | per-token posterior on the simplex; at full noise it is the positional unigram marginal m, whose argmax is "the the the , the ." (Gen. PPL 6.3, entropy 1.1) — the language "mean image". Blur = entropy / mass on the marginal. |

The layer readout we use is the paper's parameter-free probe (Eq. 2): decode the
residual stream h_i after block i with the model's own head
W_out = LayerNorm + AdaLN(c) + linear, then the same tanh soft-cap + softmax as the
model, giving delta^i (a distribution per token). For the one-step query
(tau_s, tau_t) = (0, 1), the flow map output is X_{0,1}(z) = delta_{0,1}(z) (FLM
Eq. 19), so delta^i is simultaneously the layerwise endpoint prediction and the
diagnostic clean prediction of the depth-as-time paper.

Because a single readout cannot be "looked at" like a picture, every claim below
is measured against a **ruler**: the FLM teacher's own 128-step Euler trajectory
started from the *same* z, recording D^k = D_{t_k}(x_{t_k}) at every step. We
compare (i) distances to each trajectory's own endpoint (paper Eq. 4),
(ii) student-layer vs teacher-step cross-distances, (iii) entropy / KL to the
marginal (blur), and (iv) Gen. PPL and unigram entropy of the argmax text at each
depth and time.

## 2. Results (256 noise samples unless noted)

Figures are in `results/figs/`; all numbers in `results/figs/summary.json`.

### 2.1 Progress across depth is monotone and not a readout artifact (fig1)

Normalized remaining token disagreement rho(i) with the final output:

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W_out readout | 1.00 | .96 | .95 | .91 | .88 | .85 | .81 | .55 | .46 | .38 | .27 | .16 | 0 |
| agreement w/ final | .27 | .29 | .31 | .33 | .35 | .38 | .41 | .60 | .66 | .73 | .81 | .88 | 1 |
| random probe agreement | .00 | .00 | .02 | .02 | .02 | .03 | .03 | .03 | .04 | .06 | .10 | .25 | 1 |

The random linear probe (Appendix B.1 control) sees nothing until the last two
blocks, so the progression is semantic, not "any linear map converges". The
"trained probe" of the paper (ridge fit on the final layer only) coincides with
W_out here (the head is an affine map of LayerNorm(h) for fixed conditioning), so
it is not an informative control for this model.

### 2.2 What the readout looks like: noise-seeded tokens -> marginal blur -> commit -> refine (fig2, fig7)

| depth | 0 | 1 | 2–6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| entropy [nats] | 1.08 | 3.04 | 5.1–5.3 | 1.15 | 0.69 | **0.36** | 0.46 | 0.89 | 1.35 |
| KL(readout ‖ marginal) | 4.17 | 2.79 | 1.7–1.8 | 4.53 | 4.93 | 5.42 | 5.21 | 4.86 | 4.60 |
| Gen. PPL of argmax text | 222 | 132 | 91–100 | 298 | 272 | 292 | 209 | 151 | 118 |
| unigram entropy of text | 3.87 | 3.60 | 3.40–3.49 | 4.21 | 4.23 | 4.32 | 4.23 | 4.17 | 4.14 |

(Low Gen. PPL at depths 2–6 is the usual repetition artifact: unigram entropy
3.4 vs 4.31 for real LM1B text.)

* depth 0: confident, context-free tokens that already agree 27% with the final
  output (marginal argmax: 15%, argmax z: 0.3%). h_0 = E(z) has no context, so the
  final token at a position is partly fixed by that position's own noise vector —
  the language analogue of low frequencies being inherited from the noise. See the
  noise-seeding control (2.6): a context-free noise-perturbed marginal explains
  even more (40%).
* depths 2–6: the readout is a blur toward the marginal. A shared-coefficient fit
  delta^i ≈ alpha delta^L + mu m (Eq. 5 with the noise replaced by the marginal)
  gives alpha ≈ 0.21, mu ≈ 0.77 with held-out residual R ≈ 0.22 (fig5).
* depth 7: abrupt commitment (entropy 5.3 -> 1.15, agreement 0.41 -> 0.60).
* depths 8–9 *overshoot*: sharper than the final output, then depths 10–12 soften
  while still changing tokens (agreement 0.73 -> 1) and Gen. PPL drops 292 -> 118:
  late depth does contextual repair, the analogue of fine-detail refinement.

Example (sample 1, same z):

```
L00 . is it that, [CLS] been only in the. it but capital that a who. help he - to its some ...
L06 [CLS] the it that, [CLS] of play in the property it s fall and a by. [CLS] he [CLS], are some ...
L07 [CLS] is round that, all of most in the an ve were statement and a person. [CLS] he i to ...
L10 [CLS] the legal that were all of those in the an ' s head and a half. [CLS] then i won its ...
L12 [CLS] the day that were all of those in the woman ' s head and a half. [CLS] then i saw its ...
FLM tau=0.00  [CLS] the the the, [CLS] the the. the. the the the ...
FLM tau=0.06  [CLS] the fact that, all of those in the an ' were di and a woman. [CLS] " i to ...
FLM tau=0.25  [CLS] the day that nearly all of those in the an class were black and a woman. [CLS] then i ...
FLM final     [CLS] the day that, all of those in the parking crowd were dead and a girl. [CLS] then i wonder ...
```

### 2.3 Depth maps onto teacher time, monotonically but very non-uniformly (fig3, fig4)

Matching each depth's progress (agreement with its own final output) to the
teacher's progress curve gives an effective teacher time tau*(i):

| depth | 0–6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|
| tau*(i), agreement match | ≤ 0.008 | 0.04 | 0.07 | 0.15 | 0.35 | 0.71 | 0.99 |
| tau*(i), KL-to-final match | ≤ 0.011 | 0.04 | 0.05 | 0.04 | 0.13 | 0.50 | 0.99 |

Monotone (depth as time), but not the training schedule: the first 7 blocks cover
less progress than the teacher's first 1% of tau, and the last 3 blocks do most of
the token resolution — consistent with the depth-as-time paper's observation that
the recovered depth schedule differs from the flow-matching schedule. Direct
cross-KL matching (orange in fig4) is confounded because the one-step student and
the 128-step teacher end at different sentences from the same z (token agreement
0.49), so we rely on own-endpoint normalisation as the paper does.

### 2.4 Denoise-then-renoise transfers to language as sharpen-then-blur (fig8, fig9)

FMLM is interval-conditioned on (tau_s, tau_t). Its output delta_{s,t} is a convex
average of single-time posteriors along [s, t] (FLM Prop. 3.3), so a noisy endpoint
(t < 1) has a *blurrier* target. The noise part of X_{s,t} is fixed by the
parameterisation, hence "renoise" can only appear as re-blurring of delta.
From z (tau_s = 0), entropy of delta^i_{0,t} by depth:

| tau_t | depth 6 | 7 | 8 | 9 | 10 | 11 | 12 (output) |
|---|---|---|---|---|---|---|---|
| 0.1 | 5.14 | 1.71 | 1.20 | **0.86** | 1.12 | 2.17 | 5.11 |
| 0.5 | 5.26 | 1.51 | 0.95 | **0.56** | 0.69 | 1.32 | 3.62 |
| 1.0 | 5.27 | 1.16 | 0.69 | **0.36** | 0.46 | 0.88 | 1.34 |

and the mass on the clean tokens (-log delta^i[argmax delta_{0,1}]) is lowest at
depth 10–11 (1.05 for tau_t = 0.1) and rises again at the output (1.85). The
network first resolves a clean sentence and then blurs it toward the requested
noisy endpoint, and the overshoot grows as the endpoint gets noisier — the
MeanFlow/Shortcut result of Sec. 4.2. In multi-step rollouts (2/4/8 NFEs) every
NFE restarts its own depthwise progression (nested depth-as-time, cf. Figs. 6–7 of
the paper); the first NFE (from pure noise to a noisy state) shows the clearest
sharpen-then-blur.

The FLM teacher shows the same thing inside a single denoiser NFE (fig6): its
middle blocks are sharper (entropy ≈ 0.3) than its posterior output (1.3–2.1), and
at tau = 0 blocks 0–5 decode token guesses that agree 31% with the eventual sample
before block 6 switches to the exact answer, the marginal.

### 2.5 Caveat: the depth 2–6 plateau is not a noise level

In every NFE of the multi-step rollouts — including the last one, whose input is
almost clean — depths 2–6 show the same marginal-like plateau and depth 7 the same
commitment (fig9). So the plateau is a fixed stage of the student's depthwise
computation, not "the posterior at a high noise level", and the readout at those
depths should not be read as a noisy state. The FLM teacher (same architecture,
FMLM is initialised from it) has no such plateau. The tuned lens (2.6) shows the
plateau is partly a W_out basis mismatch, while the depth-7 jump is real.

### 2.6 Controls: tuned lens and noise seeding (`results/controls/summary.json`)

Tuned lens: a separate affine probe per depth, ridge-fit from LayerNorm(h_i) to the
final logits on 8 x 16 noise samples, evaluated on 8 x 16 fresh ones.

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tuned lens: agreement w/ final | .24 | .25 | .31 | .40 | .44 | .47 | .48 | .66 | .72 | .80 | .85 | .91 | .99 |
| W_out: agreement w/ final | .27 | .29 | .31 | .34 | .35 | .38 | .41 | .59 | .66 | .72 | .80 | .88 | 1 |
| tuned lens: entropy | 6.98 | 6.64 | 5.05 | 3.80 | 3.46 | 3.55 | 3.70 | 2.02 | 1.45 | **0.87** | 0.98 | 1.18 | 1.38 |
| W_out: entropy | 1.08 | 3.03 | 5.11 | 5.11 | 5.27 | 5.32 | 5.28 | 1.15 | 0.69 | **0.36** | 0.46 | 0.90 | 1.35 |

* Under the tuned lens the readout goes **marginal-level blur -> progressively
  sharper**, like the teacher's posterior along sampling time (6.96 -> 1.13): the
  vision picture "blurry -> sharp". The confident depth-0 tokens and the 5.1–5.3
  plateau of the W_out readout are therefore partly a basis mismatch between
  mid-depth states and W_out (the tuned lens also decodes more: 0.48 vs 0.41 at
  depth 6).
* What survives both readouts: the jump at depth 7 (KL to final 2.74 -> 1.34 under
  the tuned lens) and the sharpness overshoot at depth 9 followed by re-blurring.
  (Caveat: the probe is fit by squared error on logits, so its entropy is not
  calibrated like a cross-entropy-trained lens.)

Noise seeding: a purely position-wise, context-free rule
argmax_v [log m_l(v) + beta z_l(v)] (marginal perturbed by that position's own
noise, i.e. Gumbel-max-like) predicts the final token at

| beta | 0 | 0.1 | 0.3 | 0.5 | 1 | 2 | 5 |
|---|---|---|---|---|---|---|---|
| FMLM 1-step output | .156 | .163 | .197 | .231 | .302 | **.404** | .184 |
| FLM 128-step output (same z) | .133 | .139 | .169 | .199 | .252 | **.329** | .162 |

(depth-0 W_out readout: .269 / .213). So about 40% of the one-step output tokens
(33% for the 128-step teacher) are fixed by each position's own noise vector
without any context — the language counterpart of the noise fixing the coarse
layout of an image. This is the "noise component" that a language readout carries,
and it is visible already at depth 0.

### 2.7 Toy with an exact posterior (`toy/toy.py`, `toy/toy_plots.py`)

FLM Fig. 3 toy (`cities`: "new york" / "san diego", V = 4, length 2) and an 8-mode
analogue (`ring8`, sentences (k, k), V = 8). A small FLM (8 blocks, width 128) is trained,
an FMLM is distilled with the semigroup loss from either the exact denoiser
D*_t(x) = sum_j softmax_j(t <x, X_j> / (1-t)^2) X_j or the trained FLM, and the
one-step FMLM is decoded block by block.

* cities, exact teacher: endpoint agreement with the exact flow 0.996; fraction of
  valid sentences by block .14, .22, .30, .46, .58, .85, .96, .99, 1.00. Early blocks
  leave the segment between the two sentences that the exact posterior never
  leaves: they often fix one token first (points along the square's edges).
* ring8 is finished by block 2: too easy to use the depth.
* Noisy endpoint 0 -> tau_t = 0.75: KL(clean || readout) is lowest at block 6
  (0.10) and rises to 0.24 at the output, which matches the exact target
  (readout entropy 0.235 at block 6 vs 0.471 at the output, exact 0.465). This is
  denoise-then-renoise against a known ground truth. The FLM-teacher variant agrees.

### 2.8 State matching: which flow time does each block's state equal? (`probe_statematch.py`)

No time is fed to individual blocks. Along the flow from z, x_t = (1-t) z + t delta_{0,t}(z),
so block i placed at time t reproduces x_t iff delta^i = delta_{0,t}; we take
t* = argmin_t ||delta^i - delta_{0,t}|| (distance in delta space; the state distance
carries a factor t and would always pick t = 0).

* Toy cities (exact flow as ruler): tau* = 0.32, 0.44, 0.60, 0.82, 0.97, 0.99, ...;
  a gradual sweep over blocks 0–4, mismatch falling monotonically.
* LM1B, self ruler (FMLM's own delta_{0,tau}), 128 samples:
  tau* = .06 | .06 .03 .03 .03 .03 .06 | .50 .56 .88 | 1.0 1.0 1.0.
  Teacher ruler (FLM 128-step trajectory): .21 | .16 .11 .12 .13 .14 .16 | .79 .91 .98 | .98 ...
  So two regimes and a jump, not a sweep. Blocks 1–6 sit at the start of the flow, and
  block 7 jumps to mid/late time. The mismatch is largest at blocks 7–9, whose distance
  curves are almost flat: the committed, over-sharpened stage lies on no point of the trajectory.

### 2.9 Block diagnostics for weight tying (`probe_blocks.py`)

Block substitution: replace block i by block j and measure output agreement / KL.

* LM1B, blocks 2–5 are near-interchangeable (agreement 0.83–0.89, KL 0.16–0.4).
  This is the blur plateau.
* LM1B, block 6 is irreplaceable. Anything in its slot gives about 0.5 agreement,
  and block 6 in another slot gives KL 10–15. It is the transition block right
  before the regime jump.
* LM1B, blocks 7–10 are partly interchangeable (0.6–0.9), and blocks 0 and 11 are unique.
* Toy cities: all blocks except 0 and 7 are interchangeable (agreement ≥ 0.94).

### 2.10 Making depth explicit: weight-tied FMLM (paper Sec. 5; `compress.py`)

The embedding, time embedders and head are copied from the teacher and frozen. n shared
blocks are applied K times. An internal clock (k/K, (k+1)/K) is embedded and added to the
AdaLN conditioning. The loss is the paper's Eq. 7: normalized hidden matching plus
endpoint KL. Training is data-free (z only). Warm start: each shared block starts from
the teacher block with the lowest initial loss. The middle block (6) is the worst choice
on LM1B: its initial loss is 3e5 against 6 for the best block, because of the
irreplaceable block above.

Toy cities, 6000 steps (teacher 2.51M params):

| variant | params | agree w/ teacher | KL | valid |
|---|---|---|---|---|
| tied1 (1 block + clock) | 0.53M | 0.997 | 3e-4 | 0.994 |
| tied1_noclock | 0.43M | 0.996 | 4e-4 | 0.999 |
| tied2_alt | 0.82M | 0.997 | 1e-4 | 0.994 |
| tied2_split | 0.82M | 0.996 | 1e-4 | 0.998 |
| tied2_alt, K = 4 | 0.82M | 0.999 | 1e-4 | 0.996 |

The toy is too easy to separate clock from no clock.

LM1B, 150k steps, batch 32, 256 one-step samples. Teacher blocks are 12 × 7.68M = 92.1M;
the teacher reaches Gen. PPL 115.3 and entropy 4.14.

| variant | block params | applications | agree w/ teacher | KL | Gen. PPL | entropy |
|---|---|---|---|---|---|---|
| tied_struct (groups 0\|1\|2-5\|6\|7-10\|11) | 46.2M | 12 | **0.825** | **0.32** | 115.7 | 4.11 |
| untied control (6 distinct blocks, no clock) | 46.1M | 6 | 0.808 | 0.40 | 120.3 | 4.11 |
| tied2_alt | 15.5M | 12 | 0.689 | 1.04 | 107.3 | 4.02 |
| tied2_split (block A for 0–6, B for 7–11) | 15.5M | 12 | 0.661 | 1.18 | 105.6 | 3.99 |
| tied1 (1 block + clock) | 7.8M | 12 | 0.664 | 1.18 | 101.9 | 3.99 |
| tied1_noclock | 7.7M | 12 | 0.605 | 1.52 | 90.8 | 3.90 |

OWT `tied_struct` (groups 0|1-3|4|5-10|11, 38.5M, batch 4, 128 samples): agreement 0.812,
KL 0.28, Gen. PPL 171.5 / entropy 5.06 (teacher 169.7 / 5.17).

* Sharing within the phases found by block substitution halves the block parameters
  (2.4× on OWT) at teacher-level one-step quality.
* One or two shared blocks plateau at 0.66–0.69 agreement. Their lower Gen. PPL comes
  with lower entropy, i.e. more repetitive text. Splitting the two blocks at the regime
  jump is no better than alternating them.
* The clock helps, as in the paper's App. C.4: 0.66 vs 0.61 agreement, and the clock-free
  run collapsed once at 50k steps (agreement 0.25).
* At 6 blocks reuse itself adds little. The untied 6-block control, at half the compute,
  nearly matches tied_struct (+0.02 agreement and −0.08 KL for reuse).

**Full protocol and the parameter curve** (`eval_compress.py`: 1,024 one-step samples × 3
seeds, as in the FLM paper; seed std ≤ 0.002 agreement, ≈ 1 Gen. PPL). Teacher: Gen. PPL
116.8 ± 1.2, entropy 4.146. Structured groupings use `tied_struct --groups`, all with clock,
12 applications.

| distinct blocks | shared + clock (12 applications) | agree | KL | Gen. PPL / entropy | untied (N applications) agree | KL | Gen. PPL / entropy |
|---|---|---|---|---|---|---|---|
| 1 (7.7M) | A×12 | 0.660 | 1.19 | 101.8 / 3.99 | 0.486 | 2.17 | 105.2 / 3.75 |
| 2 (15.5M) | S(0–5) B6 S(7–11) | **0.750** | 0.65 | 123.6 / 4.08 | 0.621 | 1.26 | 172.9 / 4.09 |
| 2 (15.5M) | A B A B … / A×7 B×5 | 0.686 / 0.657 | 1.04 / 1.19 | 106.7 / 4.02, 104.0 / 3.99 | | | |
| 3 (23M) | S(0–5) B6 T(7–11) | 0.761 | 0.56 | 124.7 / 4.09 | 0.712 | 0.79 | 133.7 / 4.07 |
| 4 (31M) | B0 S(1–5) B6 T(7–11) | 0.766 | 0.57 | 125.1 / 4.09 | 0.750 | 0.62 | 129.3 / 4.09 |
| 6 (46M) | B0 B1 S×4 B6 T×4 B11 | 0.824 | 0.31 | 115.2 / 4.11 | 0.805 | 0.39 | 119.3 / 4.11 |
| 1, no clock | A×12 | 0.602 | 1.54 | 90.1 / 3.90 | | | |

* **Reuse is a real parameter compression at small block counts.** The gap to an untied
  network of equal size is +0.17 agreement at 1 block, +0.13 at 2 blocks and +0.05 at 3 blocks,
  then about +0.02 at 4–6 blocks.
* **Training seeds** (`results/compress_eval/lm1b_seeds.json`; seeds 0, 1, 2, each evaluated under
  the full protocol). Agreement is 0.751 ± 0.001 for S B6 S, 0.751 ± 0.001 for untied 4 blocks and
  0.621 ± 0.001 for untied 2 blocks. Gen. PPL is 123.3 / 129.2 / 173.0 and KL 0.645 / 0.621 / 1.257.
  The parameter-compression result does not depend on the seed.
* **With 2 blocks, keeping the transition block 6 separate is what matters.** S B6 S reaches
  0.750, the level of an untied 4-block network (twice the parameters). Alternating or
  splitting two blocks reaches only 0.66–0.69.
* The 3- and 4-block groupings plateau at about 0.76, barely above 2 blocks. Merging blocks 0–5
  into one shared block is the likely bottleneck; finer early groupings are untested.
* Samples (`sample_compare.py`, `results/compress_eval/samples.json`): the same noise gives
  the same sentence skeleton in every model (noise seeding). Fewer blocks replace content words
  by function words. Entropy 4.14 → 3.99 means about 63 → 54 effective distinct tokens per
  128-token sample and a top-10-token share of 32% → 35%. Repeated bigrams go only from 5.0% to
  6.0%: this is not collapse into loops.

### 2.11 Replication on OpenWebText (FMLM-B-OWT / FLM-B-OWT)

Same 12-block DiT, GPT-2 vocabulary (50,257), length 1024. Probes are identical
(`slurm/owt_a.sh`, `slurm/owt_b.sh`; 128 samples for the lens, 64 for the rest); results
are in `results/owt/`.

| property | LM1B | OWT |
|---|---|---|
| agreement with own final, by block | .27 .29 .30 .33 .35 .38 .41 **.60** .66 .73 .81 .88 1 | .17 .25 .28 .35 .39 **.62** .68 .71 .75 .79 .84 .91 1 |
| W_out readout entropy | 1.1 3.0 5.1 5.1 5.3 5.3 5.3 **1.1** .69 **.36** .45 .89 1.35 | 3.2 2.6 3.4 3.6 3.6 **2.4** 1.8 .98 .63 **.56** .92 1.95 2.0 |
| random-probe agreement (blocks 0–9) | ≤ .06 | ≤ .12 |
| effective teacher τ (agreement match) | ≤ .008 to block 6, then .04 .07 .15 .35 .71 .99 | ≤ .007 to block 4, then .06 .12 .17 .24 .38 .59 .86 .99 |
| state matching τ*, self ruler | .06 .06 .03 .03 .03 .03 .06 \| .50 .56 .88 1 1 1 | .03 .03 .03 .06 .06 .09 .19 .44 \| 1 1 1 1 1 |
| state matching τ*, teacher ruler | .21 .16 .11 .12 .13 .14 .16 \| .79 .91 .98 ... | .04 .08 .07 .09 .12 .37 .67 .89 .98 ... |
| irreplaceable transition block | 6 | 4 (largest update norm; others in slot 4: ≤ .6 agreement) |
| interchangeable groups (swap agreement) | 2–5 (.83–.89), 7–10 (.6–.9) | 1–3 (.57–.73), 5–10 (.76–.88) |
| τ_t = 0.1 endpoint: min −log δ^i[clean] vs output | 1.05 (block 11) vs 1.85 | 1.38 (block 10) vs 2.37 |
| noise seeding, β = 2 (student / teacher) | .40 / .33 | .42 / .30 |
| student vs teacher endpoint agreement | .49 | .45 |

The qualitative picture replicates:
* monotone progress that a random probe does not reproduce;
* a noise-seed/blur stage, an abrupt commit, over-sharpening, then a final re-blur;
* back-loaded effective time;
* sharpen-then-blur toward noisy endpoints;
* about 40% of tokens set by noise seeding.

Two things differ. The commit happens two blocks earlier (block 5 instead of block 7). The
transition is also smoother: the teacher-ruler τ* goes .12 → .37 → .67 → .89 over blocks 4–7.

### 2.12 Per-token commit: confidence-ordered, revisable decoding (`probe_commit.py`)

Commit depth of a position is the first block from which the readout's argmax stays at the
final token. We used 32,768 LM1B and 65,536 OWT positions (`results/commit/`).

| | LM1B | OWT |
|---|---|---|
| committed at block 0 / at the transition block | 18% / 22% (block 7) | 13% / 28% (block 5) |
| Spearman with commit depth: final confidence / frequency / position | 0.65 / 0.38 / 0.00 | 0.68 / 0.54 / 0.00 |
| mean commit depth, noise-seeded vs other tokens | 4.1 vs 8.7 | 3.2 vs 7.5 |
| right before commit: other candidate / seeded / marginal filler | 57% / 17% / 7% | 66% / 13% / 9% |
| commit-depth std within a sentence / overall | 4.01 / 4.03 | 3.78 / 3.78 |

* **The order is confidence.** Mean commit depth falls from block 10.6 (least confident decile)
  to 2.4 (most confident) on LM1B, and from 10.1 to 1.4 on OWT. Position plays no role. This is
  the order of confidence-based unmasking in masked diffusion samplers.
* **The block-7 "jump" is not a whole-sentence event.** About a quarter of the tokens commit
  there; the rest are seeded early or trickle in later. Commit depths vary as much within a
  sentence as across sentences.
* **Unlike masked diffusion, decisions are revised.** Before committing, a position usually
  shows another candidate, not a mask-like filler. Most positions switch their argmax 1–3 times.
* The correlation with the teacher's lock step along its 128-step trajectory (0.34 / 0.44) mostly
  vanishes within confidence quintiles (0.0–0.3). Both models decide confident tokens first.

### 2.13 Why the jump: dependency length controls the commit (`toy/toy_sweep.py`)

Toys with an exact posterior: length-8 sentences over 16 tokens, built from L/c independent
chunks of c tokens, each one of m random strings. c = 1 gives independent tokens and c = 8 a
joint mixture of m sentences. We train FLM and distill an 8-block FMLM from the exact denoiser.
Results are the mean of 3 seeds, each with a new dataset (`results/toy_sweep/`).

| config | M | endpoint | τ* at block 0 | largest τ* jump | within-sentence commit std |
|---|---|---|---|---|---|
| c = 1 (independent) | 65,536 | 0.93 | 0.30 ± 0.04 | 0.28 ± 0.10 | 2.66 ± 0.18 |
| c = 2 | 256 | 0.85 | 0.11 ± 0.01 | 0.31 ± 0.04 | 2.64 ± 0.08 |
| c = 4 | 16 | 0.88 | 0.05 ± 0.00 | 0.43 ± 0.06 | 2.18 ± 0.04 |
| c = 8, m = 4 (joint) | 4 | 0.92 | 0.02 ± 0.00 | 0.45 ± 0.10 | 1.78 ± 0.03 |
| joint, m = 2 / 16 | 2 / 16 | 0.96 / 0.83 | 0.01 / 0.06 | 0.36 / 0.54 | 1.50 / 1.89 |
| joint, m = 64 / 256 (under-fit) | 64 / 256 | 0.65 / 0.21 | | | |

* Longer dependencies remove early local decisions (τ* at block 0 falls from 0.30 to 0.02). They
  make the commit more abrupt and the positions commit more synchronously. All four measures move
  monotonically with c.
* **With 16 blocks:**
  * Independent tokens spread over the whole depth (max jump 0.12).
  * Joint sentences with 4 modes still jump, now mid-depth (blocks 8–10: .32 → .92), followed by
    refinement. The commit is a decision event, not a lack of depth.
  * With 16 modes the 8-block end jump becomes gradual at 16 blocks. "More modes, more jump" was
    a depth limit.
* Width 256 does not rescue m = 64 / 256 (endpoint 0.72 / 0.52); they are excluded.
* **Interpretation, consistent with 2.12.** Locally decidable tokens (seeded, confident) are fixed
  early. Decisions that need coordination across tokens are deferred (blur), committed together
  at a transition block, and then revised in confidence order. Images have local, continuous
  pixel dependencies and sharpen gradually.

### 2.14 Direction C: trajectory supervision across depth (`toy/toy_trajsup.py`, `trajsup.py`)

Fine-tune the one-step model so that block i's readout matches the flow at τ_i, while the
endpoint keeps its target:

    L = Σ_i w_i KL(δ_{0,τ_i}(z) ‖ δ^i(z)),   w_L = 1,  w_i = λ for i < L

* **Toy:** exact δ*_{0,τ}, uniform τ_i = i/8, 5000 steps, 3 seeds.
* **LM1B:** the target is the original model's own answer to the query 0 → τ_i. A second term,
  weighted μ = 1, keeps random (s, t) queries at states reached by one teacher jump. Training
  runs 15k steps, batch 32, lr 1e-5.
* **Control:** λ = 0, the same fine-tuning on the endpoint only.

**Toy** (`results/toy_trajsup/`; agreement with the exact endpoint, control → λ = 0.1 → λ = 1)

| config | endpoint | valid sentences when sampling from block i, λ = 0.1 (before: 0 up to block 4) |
|---|---|---|
| c = 1 | 0.974 → 0.962 → 0.924 | .46 .97 1 1 1 1 1 1 |
| c = 4 | 0.978 → 0.970 → 0.946 | 0 .04 .22 .40 .59 .79 .91 .99 |
| c = 8 | 0.979 → 0.976 → 0.964 | 0 .04 .18 .45 .72 .88 .95 .99 |
| joint, m = 2 | 0.993 → 0.992 → 0.990 | .02 .29 .70 .88 .94 .98 .99 1 |

* The schedule is matched almost exactly: mean KL in the first blocks falls from 2–5 to 0.0–0.7.
  The endpoint cost is small, and smallest for joint data, against our prediction. In the toy, the
  stall is a habit of endpoint-only training, not a necessity. With 16 blocks the same holds
  (c = 8: 0.983 → 0.977).

**LM1B** (`results/trajsup/`; endpoint 1,024 samples × 3 seeds)

| setting | endpoint Gen. PPL / entropy | agree w/ original | schedule KL, blocks 1–9 | Gen. PPL from block 8 / 10 / 11 |
|---|---|---|---|---|
| control (λ = 0) | 106.4 / 4.118 | 0.889 | 1.9 | 265 / 175 / 130 |
| λ = 0.1 | 105.4 / 4.117 | 0.872 | 0.9 | 142 / 118 / 109 |
| λ = 1 | 103.3 / 4.109 | 0.835 | 0.8 | 131 / 111 / 104 |
| λ = 1, τ = 1 at block 10 | 103.6 / 4.111 | 0.834 | 0.9 | 135 / 114 / 106 |
| λ = 1, τ = 1 at block 8 | 102.9 / 4.109 | 0.828 | 1.1 | 138 / 113 / 105 |

* **Endpoint quality matches the control.** The control itself drifted from the original model
  (116.8 / 4.146).
* **The late blocks become usable for early exit.** Block 11 equals the output (104 vs 102), and
  block 10 costs about 9 Gen. PPL. Blocks 2–6 are low-entropy filler by design, since their
  targets are early, blurred times.
* **The mismatch halves but does not vanish.** Forcing the finish to block 8 or 10 does not move
  it: block-8 quality is the same as with the uniform schedule, and the middle blocks miss their
  targets (KL 1.4–1.7 at blocks 5–6). Unlike the toys, the late commit on LM1B appears to need the
  depth, at least within this budget.
* **For compute savings, distillation into a shallower network is better.** The untied 6-block
  student reaches Gen. PPL 119 at half the compute, versus 138 for exiting at block 8 (2/3 of the
  compute).

### 2.15 Early tokens become context for later ones (`probe_dependency.py`)

Dependency by intervention: resample the noise at position j only and measure the KL of the
one-step output at position i, D_{j→i}. We used 32 LM1B / 8 OWT samples, all positions j, and
2 resamples each (`results/dependency/`).

| | LM1B | OWT |
|---|---|---|
| pairs at equal distance: dependency vs. commit at similar depth (Spearman) | −0.07 … 0.00 | −0.17 … −0.08 |
| influence earlier → later vs. later → earlier, distance 1 / 2–3 / 4–16 | 2.25× / 2.95× / 3.04× | 2.50× / 2.79× / 3.21× |
| commit depth vs. dependence on the other positions (Spearman) | 0.65 | 0.62 |
| commit earlier when the own noise matters more (Spearman) | 0.30 | 0.39 |

* Dependent tokens do not commit together; they commit in order. A token that commits early
  moves later-committing tokens about three times more than the reverse, at every distance.
  Within one forward pass, later decisions are conditioned on earlier ones, as masked diffusion
  does across sampling steps. The toy's synchronous commit (2.13) belongs to symmetric, fully
  joint data. Real language dependencies are asymmetric and produce an order instead.

Token classes (share of tokens, mean commit depth):

| class | LM1B | OWT |
|---|---|---|
| punctuation / special | 20%, 5.4 (30% at block 0) | 13%, 3.9 |
| function word | 46%, 5.9 (26% at block 0) | 43%, 4.5 |
| content word | 31%, **8.7** (38% at block 7) | 30%, **7.6** (46% at block 5) |
| word piece / number | 3%, 8.7–9.1 | 14%, 6.8–7.0 |

The skeleton (punctuation and function words) comes first and content words come at the
transition block. Refinement follows.

**Multi-step sampling** (4 steps). Within every step, commit depth follows confidence: Spearman
0.60 / 0.74 / 0.72 / 0.68 on LM1B and 0.62–0.75 on OWT. The share committed at block 0 grows
with the step (18% → 55%) because more tokens arrive decided. The share committing at the
transition block stays about 23% in every step: it is a fixed stage of the network.

## 3. Verdict so far

* **Yes, with qualifications.** Progress toward the one-step output is monotone in
  depth under the model's own head, a random probe does not reproduce it, the
  layerwise text visibly passes through sentences similar to the teacher's
  intermediate predictions from the same noise, and depth maps monotonically onto
  teacher time.
* **The strongest transfer is the endpoint dependence**: for noisy endpoints the
  network sharpens to a clean sentence mid-depth and then blurs toward the
  requested two-time denoiser — the language form of denoise-then-renoise.
* **Not a smooth "noise level per layer"**: the recovered depth schedule is
  extremely back-loaded (7/12 blocks before much teacher-measurable progress), and
  the W_out plateau at depths 2–6 appears regardless of input noise. With a tuned
  lens the readout becomes a monotone blur -> sharp sequence, but the abrupt
  commitment at depth 7 remains.
* **The language "noise" is visible**: ~40% of output tokens are set position-wise
  by the input noise through a noise-perturbed marginal (Gumbel-max-like) rule.
* The paper's two-component (signal, Gaussian noise) mixing fit does not transfer
  as-is: delta lives on the simplex and has no linear z component (sigma ≈ 0). The
  natural "noise" of a language posterior is the marginal m (blur), and the
  (delta^L, m) fit is good only on the plateau (R ≈ 0.22), poor elsewhere (R ≈ 0.4–1.0).
* **Layer reuse works only within phases.** State matching shows two regimes and a jump
  rather than a gradual sweep (2.8). Consistently, one clocked shared block cannot replace
  the 12 blocks (0.66 agreement), but sharing within the blur and refine phases can, at half
  the block parameters. An untied network with the same 6 blocks comes close, so in FMLM the
  depth is better described as a few distinct stages than as one repeated time step.

## 4. Next steps

1. (done: 2.7, 2.8) Toy with an exact posterior; state matching.
2. (done: 2.10) Sec. 5 weight tying and the parameter curve. Open: tied blocks with fewer
   applications (K < 12); finer early groupings for 3–4 blocks.
3. (done: 2.11) OWT replication, including tied_struct.
4. (done: 2.12, 2.13) Per-token commit analysis; toy sweep over dependency length.
5. (done: 2.15) Token dependency, token classes, multi-step commit order.
6. (done: 2.14) Direction C, trajectory supervision. Open: longer fine-tuning on LM1B, and
   whether a full FMLM distillation with the depth loss from the start can move the commit
   earlier.

## 5. Reproduce

```bash
sbatch depth_as_time/slurm/setup_env.sbatch          # once: venv on top of NGC pytorch 26.06
sbatch --job-name=dat-onestep   depth_as_time/slurm/run.sbatch python probe_onestep.py --out results/onestep
sbatch --job-name=dat-multistep depth_as_time/slurm/run.sbatch python probe_multistep.py --out results/multistep
sbatch --job-name=dat-controls  depth_as_time/slurm/run.sbatch python probe_controls.py --out results/controls
PYTHONPATH=/ddn/scratch/nyu/awg4562/tools/analysislib python depth_as_time/analyze.py
```

Each probe takes 1–3 minutes on one GB200 (`beta_test`, 1 GPU; well under 1 SU).

| file | what |
|---|---|
| `lens.py` | backbone loading, per-block hidden states, own-head readout, FLM/FMLM samplers |
| `probe_onestep.py` | one-step lens, teacher ruler, cross-matching, probes, mixing-fit Gram stats, Gen. PPL |
| `probe_multistep.py` | endpoint sweep (denoise-then-renoise) and multi-step rollouts |
| `probe_controls.py` | tuned lens and noise-seeding controls |
| `analyze.py` | figures + `summary.json` |

Toy, state matching, block diagnostics and weight tying:

```bash
sbatch --job-name=dat-toy        depth_as_time/slurm/run.sbatch python toy/toy.py --dataset cities
sbatch --job-name=dat-statematch depth_as_time/slurm/run.sbatch python probe_statematch.py
sbatch --job-name=dat-stageA     depth_as_time/slurm/run.sbatch bash slurm/stage_a.sh   # blocks + toy tying
sbatch depth_as_time/slurm/stage_b.sbatch                                              # LM1B tying, 1 node x 4 GPUs
```

| file | what |
|---|---|
| `toy/toy.py`, `toy/toy_plots.py` | toy datasets with an exact posterior, FLM/FMLM training, probes, figures |
| `probe_statematch.py` | block-to-flow-time matching in state space (teacher and self rulers) |
| `probe_blocks.py` | residual norms, update similarity, block substitution matrix |
| `compress.py` | weight-tied FMLM with an internal clock (paper Sec. 5), data-free distillation |
| `artifact/` | interactive results page (`export_toy.py`, `export_lm1b_match.py`, `build.py`) |
