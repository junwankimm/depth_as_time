"""Depth-as-time probe for one-step FMLM generation (s=0 -> t=1).

For each noise z ~ N(0, I) in R^{L x V}:
  * Student (FMLM, one NFE with (tau_s, tau_t) = (0, 1)): decode every depth
    i = 0..12 with the model's own head -> delta^i (a per-token distribution;
    the one-step flow map output is X_{0,1}(z) = delta_{0,1}(z), Eq. 19).
  * Teacher (FLM, K Euler steps from the SAME z): record the denoiser output
    D^k = D_{t_k}(x_{t_k}) (the "x1-prediction" along sampling time) and the
    state x_{t_k} itself.
  * Teacher per-NFE depth lens at a few sampling times (nested depth-as-time).
  * Readout controls: trained ridge probe (fit on the final layer only) and a
    random probe (Appendix B.1 of the depth-as-time paper).
  * Gram statistics for the depthwise mixing fit (Eq. 5 / Appendix B.2).

All per-sample statistics are saved to an .npz file; `analyze.py` turns them
into figures and the report tables.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers

import lens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--student', default='FMLM-B-LM1B')
    p.add_argument('--teacher', default='FLM-B-LM1B')
    p.add_argument('--tokenizer', default='google-bert/bert-base-uncased')
    p.add_argument('--ppl_model', default='openai-community/gpt2-large')
    p.add_argument('--n_samples', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--teacher_steps', type=int, default=128)
    p.add_argument('--cross_every', type=int, default=2,
                   help='compare student layers with every n-th teacher step')
    p.add_argument('--nested_steps', type=str, default='0,32,64,96,112,127',
                   help='teacher steps at which to run the per-NFE depth lens')
    p.add_argument('--n_text', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='depth_as_time/results/onestep')
    return p.parse_args()


# ----------------------------------------------------------------------------
# Per-token statistics. All inputs are log-probabilities of shape (B, L, V).
# Each function returns a per-sample mean over token positions, shape (B,).
# ----------------------------------------------------------------------------

def entropy(logp):
    return -(logp.exp() * logp).sum(-1).mean(-1)


def max_prob(logp):
    return logp.max(-1).values.exp().mean(-1)


def kl(logp, logq):
    """KL(p || q) per token, averaged over positions."""
    return (logp.exp() * (logp - logq)).sum(-1).mean(-1)


def l2(logp, logq):
    """Euclidean distance between the probability tensors (whole sequence)."""
    return (logp.exp() - logq.exp()).pow(2).sum((-1, -2)).sqrt()


def agree(tok_a, tok_b):
    return (tok_a == tok_b).float().mean(-1)


def nll_of(logp, tokens):
    """-log p(token) averaged over positions: how much mass is on `tokens`."""
    return -logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).mean(-1)


def gram_stats(target, feats):
    """Per-sample normal-equation pieces for target ~ sum_j beta_j feats_j.

    Returns (G, r, yy, base) with G (B,F,F), r (B,F), yy (B,), where the sums
    run over all positions and vocabulary entries; `base` = ||target - feats_0||^2
    (the "predict the final output" baseline of Appendix B.2).
    """
    Fn = len(feats)
    G = torch.stack([torch.stack([(feats[a] * feats[b]).sum((-1, -2))
                                  for b in range(Fn)], -1)
                     for a in range(Fn)], -2)
    r = torch.stack([(feats[a] * target).sum((-1, -2)) for a in range(Fn)], -1)
    yy = target.pow(2).sum((-1, -2))
    base = (target - feats[0]).pow(2).sum((-1, -2))
    return G, r, yy, base


class GenPPL:
    """Generative perplexity under GPT-2 Large, same protocol as
    metrics.Metrics.record_generative_perplexity (retokenize, chunk, ignore
    tokens after the first EOS), but keeping one eval model in memory."""

    def __init__(self, name, device, batch_size=32):
        self.tok = transformers.AutoTokenizer.from_pretrained(name)
        self.tok.pad_token = self.tok.eos_token
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            name).eval().to(device)
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, texts, max_length=128):
        enc = self.tok(texts, return_tensors='pt', return_token_type_ids=False,
                       return_attention_mask=True, truncation=True,
                       padding=True, max_length=max_length)
        ids, mask = enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device)
        tot_nll, tot_n = 0.0, 0.0
        for i in range(0, ids.shape[0], self.batch_size):
            x, m = ids[i:i + self.batch_size], mask[i:i + self.batch_size]
            logits = self.model(x, attention_mask=m)[0].transpose(-1, -2)
            nlls = F.cross_entropy(logits[..., :-1], x[..., 1:], reduction='none')
            first_eos = (x == self.tok.eos_token_id).cumsum(-1) == 1
            token_mask = x != self.tok.eos_token_id
            valid = (first_eos[..., 1:] + token_mask[..., 1:]).float()
            tot_nll += (nlls * valid).sum().item()
            tot_n += valid.sum().item()
        return float(np.exp(tot_nll / tot_n))


def unigram_entropy(tokens):
    """Per-sample unigram entropy (metrics.Metrics.record_entropy)."""
    out = []
    for sample in tokens:
        _, counts = torch.unique(sample, return_counts=True, sorted=False)
        out.append(torch.special.entr(counts.float() / counts.sum()).sum().item())
    return float(np.mean(out))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda'
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    student = lens.load_backbone(args.student, device)
    teacher = lens.load_backbone(args.teacher, device)
    V = student.vocab_size
    Lseq = student.config.model.length
    depth = len(student.blocks)
    lut = lens.TimeLUT(V)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    K = args.teacher_steps
    cross_steps = list(range(0, K, args.cross_every))
    if cross_steps[-1] != K - 1:
        cross_steps.append(K - 1)
    nested_steps = [int(s) for s in args.nested_steps.split(',')]

    n_batches = args.n_samples // args.batch_size
    N = n_batches * args.batch_size
    nL, nK, nC = depth + 1, K, len(cross_steps)
    S = {  # per-sample statistics
        # student, per depth i
        's_ent': np.zeros((N, nL)), 's_maxp': np.zeros((N, nL)),
        's_agree': np.zeros((N, nL)), 's_kl': np.zeros((N, nL)),
        's_l2': np.zeros((N, nL)), 's_kl_marg': np.zeros((N, nL)),
        's_nll_final': np.zeros((N, nL)), 's_agree_teacher': np.zeros((N, nL)),
        # agreement with argmax(z) (raw noise) and with the marginal's argmax
        's_agree_noise': np.zeros((N, nL)), 's_agree_marg': np.zeros((N, nL)),
        # teacher, per sampling step k (denoiser output D^k and state x^k)
        't_ent': np.zeros((N, nK)), 't_maxp': np.zeros((N, nK)),
        't_agree': np.zeros((N, nK)), 't_kl': np.zeros((N, nK)),
        't_l2': np.zeros((N, nK)), 't_kl_marg': np.zeros((N, nK)),
        't_state_agree': np.zeros((N, nK)), 't_tau': np.zeros(nK), 't_t': np.zeros(nK),
        # student layer i vs teacher step k (same z)
        'x_kl': np.zeros((N, nL, nC)), 'x_kl_rev': np.zeros((N, nL, nC)),
        'x_agree': np.zeros((N, nL, nC)),
        # nested: teacher depth lens at a few NFEs
        'n_kl': np.zeros((N, len(nested_steps), nL)),
        'n_ent': np.zeros((N, len(nested_steps), nL)),
        'n_agree_final': np.zeros((N, len(nested_steps), nL)),
        # readout controls (distance to the probe's own final-layer output)
        'c_ridge_kl': np.zeros((N, nL)), 'c_rand_kl': np.zeros((N, nL)),
        'c_ridge_agree': np.zeros((N, nL)), 'c_rand_agree': np.zeros((N, nL)),
        # student <-> teacher endpoint agreement
        'st_final_agree': np.zeros(N),
    }
    # mixing fit statistics: prob space [delta^L, m, z], logp space [logp^L, z, 1]
    mix = {k: [] for k in ['p_G', 'p_r', 'p_yy', 'p_base', 'l_G', 'l_r', 'l_yy', 'l_base',
                           'tp_G', 'tp_r', 'tp_yy', 'tp_base']}
    s_tokens = np.zeros((N, nL, Lseq), dtype=np.int32)
    t_tokens = np.zeros((N, nC, Lseq), dtype=np.int32)
    t_final_tokens = np.zeros((N, Lseq), dtype=np.int32)
    ridge = None
    t0 = time.time()

    for b in range(n_batches):
        sl = slice(b * args.batch_size, (b + 1) * args.batch_size)
        B = args.batch_size
        z = torch.randn((B, Lseq, V), device=device, dtype=torch.float32)

        # ---------------- student: one NFE, all depths -----------------
        s_logp, hs, c = lens.layer_logprobs(student, z, 0.0, 1.0)
        if b == 0:  # the depth-L readout must equal the model output
            ref = lens.model_logprobs(student, z, 0.0, 1.0)
            err = (ref - s_logp[-1]).abs().max().item()
            print(f'[check] |readout_L - model_output|_max = {err:.3e}', flush=True)
            assert err < 1e-3, err
        s_tok = [lp.argmax(-1) for lp in s_logp]
        y_s = s_tok[-1]
        for i, lp in enumerate(s_logp):
            S['s_ent'][sl, i] = entropy(lp).cpu().numpy()
            S['s_maxp'][sl, i] = max_prob(lp).cpu().numpy()
            S['s_agree'][sl, i] = agree(s_tok[i], y_s).cpu().numpy()
            S['s_kl'][sl, i] = kl(s_logp[-1], lp).cpu().numpy()
            S['s_l2'][sl, i] = l2(s_logp[-1], lp).cpu().numpy()
            S['s_nll_final'][sl, i] = nll_of(lp, y_s).cpu().numpy()
            s_tokens[sl, i] = s_tok[i].cpu().numpy()

        # ---------------- readout controls (Appendix B.1) ---------------
        ln = [F.layer_norm(h.float(), [h.shape[-1]]) for h in hs]
        if ridge is None:
            X = torch.cat([ln[-1].reshape(-1, ln[-1].shape[-1]),
                           torch.ones(B * Lseq, 1, device=device)], -1)
            Y = lens.readout_logits(student, hs[-1], c).reshape(-1, V)
            lam = 1e-3 * (X * X).sum() / X.shape[1]
            ridge = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=device),
                                       X.T @ Y)
            rand = torch.randn_like(ridge) * ridge.std()
            del X, Y

        def probe(W, x):
            x = torch.cat([x, torch.ones(*x.shape[:-1], 1, device=device)], -1)
            return lens._logprobs(x @ W)
        for name, W in [('ridge', ridge), ('rand', rand)]:
            outs = [probe(W, x) for x in ln]
            fin_tok = outs[-1].argmax(-1)
            for i, lp in enumerate(outs):
                S[f'c_{name}_kl'][sl, i] = kl(outs[-1], lp).cpu().numpy()
                S[f'c_{name}_agree'][sl, i] = agree(lp.argmax(-1), fin_tok).cpu().numpy()
            del outs
        del ln, hs

        # ---------------- teacher: K Euler steps from the same z ---------
        rec = {}

        def callback(k, tau_k, t_k, x_k, logp_k):
            S['t_tau'][k], S['t_t'][k] = tau_k, t_k
            rec.setdefault('state_tok', {})[k] = x_k.argmax(-1)
            if k == 0:
                rec['marg'] = logp_k.clone()
            if k in cross_steps:
                ci = cross_steps.index(k)
                tok_k = logp_k.argmax(-1)
                t_tokens[sl, ci] = tok_k.cpu().numpy()
                for i, lp in enumerate(s_logp):
                    S['x_kl'][sl, i, ci] = kl(logp_k, lp).cpu().numpy()
                    S['x_kl_rev'][sl, i, ci] = kl(lp, logp_k).cpu().numpy()
                    S['x_agree'][sl, i, ci] = agree(s_tok[i], tok_k).cpu().numpy()
            # D^k statistics that need the final output are done afterwards;
            # keep only what is cheap here.
            rec.setdefault('logp', {})[k] = logp_k.to(torch.bfloat16)
            if k in nested_steps:
                ni = nested_steps.index(k)
                n_logp, _, _ = lens.layer_logprobs(teacher, x_k, tau_k)
                rec.setdefault('nested', {})[ni] = [lp.argmax(-1) for lp in n_logp]
                for i, lp in enumerate(n_logp):
                    S['n_kl'][sl, ni, i] = kl(n_logp[-1], lp).cpu().numpy()
                    S['n_ent'][sl, ni, i] = entropy(lp).cpu().numpy()
                del n_logp

        y_t = lens.flm_trajectory(teacher, lut, z, K, callback)
        t_final_tokens[sl] = y_t.cpu().numpy()
        S['st_final_agree'][sl] = agree(y_s, y_t).cpu().numpy()
        marg = rec['marg']
        logp_T = rec['logp'][K - 1].float()
        for k in range(K):
            lp = rec['logp'][k].float()
            S['t_ent'][sl, k] = entropy(lp).cpu().numpy()
            S['t_maxp'][sl, k] = max_prob(lp).cpu().numpy()
            S['t_agree'][sl, k] = agree(lp.argmax(-1), y_t).cpu().numpy()
            S['t_kl'][sl, k] = kl(logp_T, lp).cpu().numpy()
            S['t_l2'][sl, k] = l2(logp_T, lp).cpu().numpy()
            S['t_kl_marg'][sl, k] = kl(lp, marg).cpu().numpy()
            S['t_state_agree'][sl, k] = agree(rec['state_tok'][k], y_t).cpu().numpy()
        for ni in rec.get('nested', {}):
            toks = rec['nested'][ni]
            for i in range(nL):
                S['n_agree_final'][sl, ni, i] = agree(toks[i], y_t).cpu().numpy()
        z_tok, marg_tok = z.argmax(-1), marg.argmax(-1)
        for i, lp in enumerate(s_logp):
            S['s_agree_noise'][sl, i] = agree(s_tok[i], z_tok).cpu().numpy()
            S['s_agree_marg'][sl, i] = agree(s_tok[i], marg_tok).cpu().numpy()
            S['s_kl_marg'][sl, i] = kl(lp, marg).cpu().numpy()
            S['s_agree_teacher'][sl, i] = agree(s_tok[i], y_t).cpu().numpy()

        # ---------------- mixing-fit statistics (Eq. 5) ------------------
        p_final, p_marg = s_logp[-1].exp(), marg.exp()
        out = [gram_stats(lp.exp(), [p_final, p_marg, z]) for lp in s_logp]
        for key, j in [('p_G', 0), ('p_r', 1), ('p_yy', 2), ('p_base', 3)]:
            mix[key].append(torch.stack([o[j] for o in out], 1).cpu().numpy())
        ones = torch.ones_like(z)
        out = [gram_stats(lp, [s_logp[-1], z, ones]) for lp in s_logp]
        for key, j in [('l_G', 0), ('l_r', 1), ('l_yy', 2), ('l_base', 3)]:
            mix[key].append(torch.stack([o[j] for o in out], 1).cpu().numpy())
        pT = logp_T.exp()
        out = [gram_stats(rec['logp'][k].float().exp(), [pT, p_marg, z]) for k in cross_steps]
        for key, j in [('tp_G', 0), ('tp_r', 1), ('tp_yy', 2), ('tp_base', 3)]:
            mix[key].append(torch.stack([o[j] for o in out], 1).cpu().numpy())
        del rec, s_logp, out, ones
        torch.cuda.empty_cache()
        print(f'batch {b + 1}/{n_batches}  student/teacher final agreement '
              f'{S["st_final_agree"][sl].mean():.3f}  ({time.time() - t0:.0f}s)', flush=True)

    # ---------------- generative perplexity across depth / time ---------
    del student, teacher
    torch.cuda.empty_cache()
    gen = GenPPL(args.ppl_model, device)
    text = {'student': [], 'teacher': []}
    quality = {'student_layer': [], 'teacher_step': [], 'teacher_final': None}
    for i in range(nL):
        toks = torch.from_numpy(s_tokens[:, i]).long()
        texts = tokenizer.batch_decode(toks)
        quality['student_layer'].append(
            {'layer': i, 'gen_ppl': gen(texts, Lseq), 'entropy': unigram_entropy(toks)})
        text['student'].append(texts[:args.n_text])
    for ci, k in enumerate(cross_steps):
        if ci % 4 and k != K - 1:
            continue
        toks = torch.from_numpy(t_tokens[:, ci]).long()
        texts = tokenizer.batch_decode(toks)
        quality['teacher_step'].append(
            {'step': k, 'tau': float(S['t_tau'][k]), 'gen_ppl': gen(texts, Lseq),
             'entropy': unigram_entropy(toks)})
    toks = torch.from_numpy(t_final_tokens).long()
    texts = tokenizer.batch_decode(toks)
    quality['teacher_final'] = {'gen_ppl': gen(texts, Lseq), 'entropy': unigram_entropy(toks)}
    text['teacher_final'] = texts[:args.n_text]
    text['teacher'] = {int(k): tokenizer.batch_decode(
        torch.from_numpy(t_tokens[:args.n_text, ci]).long())
        for ci, k in enumerate(cross_steps)}
    for row in quality['student_layer']:
        print(f"student layer {row['layer']:2d}: gen PPL {row['gen_ppl']:10.2f}  "
              f"entropy {row['entropy']:.2f}")
    for row in quality['teacher_step']:
        print(f"teacher step {row['step']:4d} (tau={row['tau']:.2f}): "
              f"gen PPL {row['gen_ppl']:10.2f}  entropy {row['entropy']:.2f}")
    print('teacher final:', quality['teacher_final'])

    np.savez_compressed(
        os.path.join(args.out, 'stats.npz'),
        cross_steps=np.array(cross_steps), nested_steps=np.array(nested_steps),
        s_tokens=s_tokens, t_tokens=t_tokens, t_final_tokens=t_final_tokens,
        **S, **{k: np.concatenate(v, 0) for k, v in mix.items()})
    with open(os.path.join(args.out, 'quality.json'), 'w') as f:
        json.dump(quality, f, indent=1)
    with open(os.path.join(args.out, 'text.json'), 'w') as f:
        json.dump(text, f, indent=1)
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)
    print(f'done in {time.time() - t0:.0f}s -> {args.out}')


if __name__ == '__main__':
    main()
