"""Direction C on LM1B/OWT: trajectory supervision of a one-step FMLM across depth.

Fine-tune a copy of the FMLM (student) against the frozen original (teacher), data-free:
    L = sum_{i=1}^{L} w_i KL(delta^T_{0, tau_i}(z) || delta^i_S(z))       tau_i = i / L
      + mu * KL(delta^T_{s,t}(x_s) || delta_S{s,t}(x_s))                     (keep other queries)
where delta^i_S is the student's readout after block i (own head, one-step query (0, 1)),
delta^T_{0,tau} the teacher's own answer to the query 0 -> tau (the "self ruler" of
probe_statematch.py), w_L = 1 (endpoint) and w_i = lam for i < L. lam = 0 is the control:
the same fine-tuning with the endpoint only. The second term keeps the multi-time flow map
on random (s, t) queries at states x_s = X^T_{0,s}(z) reached by one teacher jump.

After training: endpoint quality under the paper protocol (1,024 samples x seeds), agreement
with the original model, early-exit quality (argmax of each block's readout as the sample),
and the per-block KL to the uniform schedule.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lens  # noqa: E402
from probe_onestep import GenPPL, unigram_entropy  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='FMLM-B-LM1B')
    p.add_argument('--tokenizer', default='google-bert/bert-base-uncased')
    p.add_argument('--ppl_model', default='openai-community/gpt2-large')
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--mu', type=float, default=1.0)
    p.add_argument('--finish_block', type=int, default=0,
                   help='block at which the schedule reaches tau = 1 (0: the last block, i.e. tau_i = i / L)')
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--log_every', type=int, default=250)
    p.add_argument('--time_budget', type=float, default=0, help='minutes; stop training early to leave time for eval')
    p.add_argument('--n_eval', type=int, default=1024)
    p.add_argument('--eval_seeds', default='1,2,3')
    p.add_argument('--eval_bs', type=int, default=64)
    p.add_argument('--out', default='results/trajsup')
    p.add_argument('--tag', default='')
    return p.parse_args()


def kl(p_log, q_log):
    """Mean over batch and positions of KL(p || q) from log-probabilities."""
    return (p_log.exp() * (p_log - q_log)).sum(-1).mean()


class EMA:
    def __init__(self, model, decay):
        self.decay, self.n = decay, 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)


def main():
    args = parse_args()
    device = 'cuda'
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    teacher = lens.load_backbone(args.model, device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = copy.deepcopy(teacher).train()
    for p in student.parameters():
        p.requires_grad_(True)
    V, Lseq, nB = teacher.vocab_size, teacher.config.model.length, len(teacher.blocks)
    lut = lens.TimeLUT(V)
    fin = args.finish_block or len(teacher.blocks)
    name = f"{os.path.basename(args.model).lower()}_lam{args.lam:g}_mu{args.mu:g}" + (f'_fin{fin}' if fin != len(teacher.blocks) else '') + args.tag
    out_dir = os.path.join(HERE, args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    taus = [min(1.0, (i + 1) / fin) for i in range(nB)]  # block i+1 targets tau = (i+1)/fin, capped at 1
    w = torch.tensor([args.lam] * (nB - 1) + [1.0], device=device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    ema = EMA(student, args.ema)
    B = args.batch_size
    t0 = time.time()
    for step in range(args.steps):
        z = torch.randn((B, Lseq, V), device=device)
        with torch.no_grad():
            targets = [lens.model_logprobs(teacher, z, 0.0, tau) for tau in taus]
            # a random (s, t) query at a state reached by one teacher jump from z
            tau_s = torch.rand(B, device=device) * 0.95
            tau_t = tau_s + (1 - tau_s) * torch.rand(B, device=device)
            t_s = lut.tau_to_t(tau_s).float().view(-1, 1, 1)
            x_s = (1 - t_s) * z + t_s * lens.model_logprobs(teacher, z, torch.zeros(B, device=device), tau_s).exp()
            multi_t = lens.model_logprobs(teacher, x_s, tau_s, tau_t)
        logps, _, _ = lens.layer_logprobs_grad(student, z, 0.0, 1.0)
        kls = torch.stack([kl(tg, lp) for tg, lp in zip(targets, logps[1:])])
        loss_traj = (w * kls).sum() / w.sum()
        loss_multi = kl(multi_t, lens._logprobs(student(x_s, tau_s, tau_t))) if args.mu > 0 else torch.zeros((), device=device)
        loss = loss_traj + args.mu * loss_multi
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(student)
        stop = args.time_budget > 0 and (time.time() - t0) / 60 > args.time_budget
        if step % args.log_every == 0 or step == args.steps - 1 or stop:
            print(f'step {step} loss {loss.item():.4f} end-KL {kls[-1].item():.4f} '
                  f'mid-KL {kls[:-1].mean().item():.4f} multi {loss_multi.item():.4f} ({time.time() - t0:.0f}s)',
                  flush=True)
        if stop:
            print(f'time budget reached at step {step + 1}', flush=True)
            break
    res_steps = step + 1
    student.load_state_dict(ema.shadow)
    student.eval()
    torch.save(student.state_dict(), os.path.join(out_dir, 'student_ema.pt'))

    # ---------------- evaluation ----------------
    del opt
    torch.cuda.empty_cache()
    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    gen = GenPPL(args.ppl_model, device)
    seeds = [int(s) for s in args.eval_seeds.split(',')]
    res = {'name': name, 'args': vars(args), 'steps_done': res_steps, 'taus': taus, 'endpoint': {}, 'early_exit': {}}

    @torch.no_grad()
    def collect(seed):
        """Per block: tokens of student readouts, original-model tokens, KL to the schedule."""
        g = torch.Generator(device=device).manual_seed(seed)
        toks = [[] for _ in range(nB + 1)]
        ref, sch = [], np.zeros(nB)
        n_b = args.n_eval // args.eval_bs
        for _ in range(n_b):
            z = torch.randn((args.eval_bs, Lseq, V), device=device, generator=g)
            lps, _, _ = lens.layer_logprobs(student, z, 0.0, 1.0)
            for i, lp in enumerate(lps):
                toks[i].append(lp.argmax(-1).cpu())
            for i, tau in enumerate(taus):
                sch[i] += kl(lens.model_logprobs(teacher, z, 0.0, tau), lps[i + 1]).item() / n_b
            ref.append(lens.model_logprobs(teacher, z, 0.0, 1.0).argmax(-1).cpu())
        return [torch.cat(t) for t in toks], torch.cat(ref), sch

    for sd in seeds:
        toks, ref, sch = collect(sd)
        e = res['endpoint']
        e.setdefault('gen_ppl', []).append(gen(tok.batch_decode(toks[-1]), Lseq))
        e.setdefault('entropy', []).append(unigram_entropy(toks[-1]))
        e.setdefault('agree_original', []).append((toks[-1] == ref).float().mean().item())
        if sd == seeds[0]:
            res['kl_to_schedule'] = sch.round(4).tolist()
            ee = res['early_exit']
            for i in range(nB + 1):
                ee.setdefault('gen_ppl', []).append(gen(tok.batch_decode(toks[i]), Lseq))
                ee.setdefault('entropy', []).append(unigram_entropy(toks[i]))
                ee.setdefault('agree_own_final', []).append((toks[i] == toks[-1]).float().mean().item())
            res['samples'] = [tok.decode(t) for t in toks[-1][:3]]
    with open(os.path.join(out_dir, 'eval.json'), 'w') as f:
        json.dump(res, f, indent=1)
    m = lambda v: f'{np.mean(v):.3f} ± {np.std(v):.3f}'  # noqa: E731
    print(f"[{name}] endpoint Gen PPL {m(res['endpoint']['gen_ppl'])}, entropy {m(res['endpoint']['entropy'])}, "
          f"agree w/ original {m(res['endpoint']['agree_original'])}")
    print('  kl_to_schedule', res['kl_to_schedule'])
    print('  early-exit gen_ppl', np.round(res['early_exit']['gen_ppl'], 1).tolist())
    print('  early-exit entropy', np.round(res['early_exit']['entropy'], 3).tolist())
    print('  early-exit agree  ', np.round(res['early_exit']['agree_own_final'], 3).tolist())


if __name__ == '__main__':
    main()
