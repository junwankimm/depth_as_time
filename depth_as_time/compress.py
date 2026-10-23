"""Make depth as time explicit for FMLM (depth-as-time paper, Sec. 5 / App. C).

Replace the L layer-specific blocks of a one-step FMLM teacher with n in {1, 2}
unique blocks applied K times. Each application k gets an internal clock
interval (tau_a, tau_b) = (k/K, (k+1)/K), embedded and added to the task
conditioning (tau_s, tau_t) = (0, 1) before the AdaLN of the shared block.
The input embedding, task-time embedders and output head are copied from the
teacher and frozen; only the shared block(s) and the clock embedders train.

Objective (paper Eq. 7, with the image-space anchor replaced by its
simplex counterpart):
    L = 1/K sum_k ||h_k - H_{k s}||^2 / ||H_{k s}||^2  +  lambda KL(delta_T || delta_S)
with stride s = L/K. Training is data-free: inputs are z ~ N(0, I) only.

Variants (--variant):
  tied1        one shared block, with clock
  tied1_noclock  one shared block, no clock (App. C.4 ablation)
  tied2_alt    two blocks alternating, with clock (paper's two-block model)
  tied2_split  two blocks, block A for the first `split` applications and block B
               after (FMLM's readout changes regime at depth 7), with clock
  untied       control: K distinct blocks, each applied once (no sharing, no clock);
               with K = 6 it has as many blocks as tied_struct, and half its compute
  tied_struct  K = L; teacher depth grouped by --groups (default from the block
               substitution test: 0|1|2-5|6|7-10|11), one block per group, with clock

Warm start (--init auto): each shared block starts from the teacher block (among
the teacher depths it replaces) that gives the lowest initial loss, chosen by one
pass of coordinate search on a probe batch. A fixed middle block can be a poor
choice: in the LM1B FMLM, block 6 is the one block that cannot be substituted.
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'toy'))
import lens  # noqa: E402
import models.dit  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', default='FMLM-B-LM1B',
                   help="HF snapshot name/dir, or 'toy:<dataset>:<tag>' for a toy FMLM")
    p.add_argument('--variant', default='tied1',
                   choices=['tied1', 'tied1_noclock', 'tied2_alt', 'tied2_split', 'tied_struct', 'untied'])
    p.add_argument('--K', type=int, default=None, help='block applications (default: teacher depth)')
    p.add_argument('--split', type=int, default=7, help='tied2_split: applications using block A')
    p.add_argument('--groups', default='0|1|2-5|6|7-10|11', help='tied_struct: teacher-depth groups')
    p.add_argument('--init', default='auto', help="'auto', 'middle', or comma-separated teacher block ids")
    p.add_argument('--log_every', type=int, default=500)
    p.add_argument('--tag', default='', help='suffix for the run name (e.g. to tell tied_struct groupings apart)')
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--eval_every', type=int, default=10000)
    p.add_argument('--n_eval', type=int, default=256)
    p.add_argument('--ckpt_every', type=int, default=2000)
    p.add_argument('--ppl_model', default='openai-community/gpt2-large')
    p.add_argument('--tokenizer', default='google-bert/bert-base-uncased')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/compress')
    return p.parse_args()


# ----------------------------------------------------------------------------
# Teacher loading
# ----------------------------------------------------------------------------

def load_teacher(spec, device):
    """Returns (backbone, info). info: vocab, length, is_toy, dataset."""
    if spec.startswith('toy:'):
        import toy  # noqa: E402
        _, ds, tag = spec.split(':')
        targs = argparse.Namespace(hidden=128, n_blocks=8)
        V = toy.DATASETS[ds]['V']
        model = toy.make_dit(targs, V, 'fmlm')
        model.load_state_dict(torch.load(os.path.join(HERE, 'results', 'toy', ds, f'fmlm_{tag}.pt')))
        return model.eval(), {'vocab': V, 'length': 2, 'is_toy': True, 'dataset': ds}
    model = lens.load_backbone(spec, device)
    return model, {'vocab': model.vocab_size, 'length': model.config.model.length,
                   'is_toy': False, 'dataset': 'owt' if 'OWT' in spec else 'lm1b'}


# ----------------------------------------------------------------------------
# Student
# ----------------------------------------------------------------------------

def parse_groups(spec):
    """'0|1|2-5|6' -> [[0], [1], [2, 3, 4, 5], [6]]; a group may join ranges: '0-5,7-11|6'."""
    out = []
    for g in spec.split('|'):
        ids = []
        for r in g.split(','):
            a, _, b = r.partition('-')
            ids += list(range(int(a), int(b or a) + 1))
        out.append(ids)
    return out


class TiedFMLM(nn.Module):
    """Teacher's embedding / time embedders / head (frozen) + shared blocks."""

    def __init__(self, teacher, variant, K, split, groups='0|1|2-5|6|7-10|11', init=None):
        super().__init__()
        L = len(teacher.blocks)
        self.variant, self.K, self.split, self.depth = variant, K, split, L
        stride = L // K
        # assign[k]: shared block used by application k; cand[b]: teacher depths block b replaces
        if variant.startswith('tied1'):
            self.assign = [0] * K
        elif variant == 'tied2_alt':
            self.assign = [k % 2 for k in range(K)]
        elif variant == 'tied2_split':
            self.assign = [int(k >= self.split_k) for k in range(K)]
        elif variant == 'untied':
            self.assign = list(range(K))
        else:
            assert K == L, 'tied_struct needs K = teacher depth'
            gs = parse_groups(groups)
            assert sorted(sum(gs, [])) == list(range(L))
            self.assign = [next(b for b, g in enumerate(gs) if k in g) for k in range(K)]
        nb = max(self.assign) + 1
        self.cand = [sorted({d for k in range(K) if self.assign[k] == b for d in range(k * stride, (k + 1) * stride)})
                     for b in range(nb)]
        if variant in ('tied1', 'tied1_noclock', 'tied2_alt'):
            self.cand = [list(range(L))] * nb  # these blocks stand in for every depth
        if init is None:  # middle of each candidate range
            init = [c[len(c) // 2] if variant != 'tied2_alt' else [L // 2 - 1, L // 2][b]
                    for b, c in enumerate(self.cand)]
        self.init = list(init)
        self.vocab_embed = copy.deepcopy(teacher.vocab_embed)
        self.sigma_map = copy.deepcopy(teacher.sigma_map)
        self.sigma_map_prime = copy.deepcopy(teacher.sigma_map_prime)
        self.rotary_emb = copy.deepcopy(teacher.rotary_emb)
        self.output_layer = copy.deepcopy(teacher.output_layer)
        for m in [self.vocab_embed, self.sigma_map, self.sigma_map_prime, self.output_layer]:
            for p in m.parameters():
                p.requires_grad_(False)
        self.blocks = nn.ModuleList([copy.deepcopy(teacher.blocks[i]) for i in self.init])
        for p in self.blocks.parameters():
            p.requires_grad_(True)
        cond_dim = teacher.sigma_map.mlp[-1].out_features
        self.use_clock = not (variant.endswith('noclock') or variant == 'untied')
        if self.use_clock:
            self.clock_a = models.dit.TimestepEmbedder(cond_dim)
            self.clock_b = models.dit.TimestepEmbedder(cond_dim)
            for emb in (self.clock_a, self.clock_b):  # start as the teacher's block
                nn.init.zeros_(emb.mlp[-1].weight)
                nn.init.zeros_(emb.mlp[-1].bias)

    @property
    def split_k(self):
        # number of applications that use block A (split is in teacher-depth units)
        return max(1, min(self.K - 1, round(self.split * self.K / self.depth)))

    def forward(self, x, tau_s, tau_t, return_hidden=False, blocks=None):
        blocks = self.blocks if blocks is None else blocks
        B = x.shape[0]
        h = self.vocab_embed(x)
        t_emb = self.sigma_map(tau_s) + self.sigma_map_prime(tau_t)
        c_task = F.silu(t_emb)
        rot = self.rotary_emb(h)
        hs = []
        grid = torch.linspace(0, 1, self.K + 1, device=x.device)
        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for k in range(self.K):
                if self.use_clock:
                    c = F.silu(t_emb + self.clock_a(grid[k].expand(B)) + self.clock_b(grid[k + 1].expand(B)))
                else:
                    c = c_task
                h = blocks[self.assign[k]](h, rot, c=c, seqlens=None)
                hs.append(h)
            logits = self.output_layer(h, c=c_task)
        logp = lens._logprobs(logits.float())
        return (logp, hs) if return_hidden else logp


def distill_loss(student, teacher, z, stride, lam, blocks=None):
    """Paper Eq. 7: normalized hidden-state matching + endpoint KL. Returns (loss, lh, le)."""
    B = z.shape[0]
    with torch.no_grad():
        Hs, c = lens.hidden_states(teacher, z, 0.0, 1.0)
        target = lens.readout(teacher, Hs[-1], c)
    logp, hs = student(z, z.new_zeros(B), z.new_ones(B), return_hidden=True, blocks=blocks)
    lh = 0.0
    for k in range(student.K):
        H = Hs[(k + 1) * stride].float()
        lh = lh + ((hs[k].float() - H).pow(2).sum((-1, -2)) / H.pow(2).sum((-1, -2))).mean()
    lh = lh / student.K
    le = (target.exp() * (target - logp)).sum(-1).mean()
    return lh + lam * le, lh, le


@torch.no_grad()
def search_init(student, teacher, z, stride, lam):
    """One pass of coordinate search over warm-start blocks (min initial loss)."""
    init = list(student.init)

    def score(ids):
        return distill_loss(student, teacher, z, stride, lam, blocks=[teacher.blocks[i] for i in ids])[0].item()

    best = score(init)
    for b, cands in enumerate(student.cand):
        for j in cands:
            trial = init[:b] + [j] + init[b + 1:]
            v = score(trial)
            if v < best:
                best, init = v, trial
    return init, best


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(student, teacher, info, args, gen=None, tok=None, seed=1234):
    g = torch.Generator(device='cuda').manual_seed(seed)
    V, Lseq = info['vocab'], info['length']
    res = {'agree_teacher': [], 'kl_teacher': []}
    toks_s, toks_t = [], []
    bs = 32
    for _ in range(max(1, args.n_eval // bs)):
        z = torch.randn((bs, Lseq, V), device='cuda', generator=g)
        lt = lens.model_logprobs(teacher, z, 0.0, 1.0)
        ls = student(z, torch.zeros(bs, device='cuda'), torch.ones(bs, device='cuda'))
        ys, yt = ls.argmax(-1), lt.argmax(-1)
        res['agree_teacher'].append((ys == yt).float().mean().item())
        res['kl_teacher'].append((lt.exp() * (lt - ls)).sum(-1).mean().item())
        toks_s.append(ys.cpu())
        toks_t.append(yt.cpu())
    out = {k: float(np.mean(v)) for k, v in res.items()}
    toks_s, toks_t = torch.cat(toks_s), torch.cat(toks_t)
    if info['is_toy']:
        import toy  # noqa: E402
        valid = {tuple(s) for s in toy.DATASETS[info['dataset']]['seqs']}
        out['valid'] = float(np.mean([tuple(r) in valid for r in toks_s.tolist()]))
        out['valid_teacher'] = float(np.mean([tuple(r) in valid for r in toks_t.tolist()]))
    elif gen is not None:
        from probe_onestep import unigram_entropy
        out['gen_ppl'] = gen(tok.batch_decode(toks_s), Lseq)
        out['entropy'] = unigram_entropy(toks_s)
        out['gen_ppl_teacher'] = gen(tok.batch_decode(toks_t), Lseq)
        out['entropy_teacher'] = unigram_entropy(toks_t)
        out['sample'] = tok.decode(toks_s[0][:60])
    return out


class EMA:
    def __init__(self, params, decay):
        self.params = [p for p in params if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]
        self.decay = decay
        self.n = 0

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))  # EMA warmup, so short runs are not stuck at init
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach(), alpha=1 - d)

    def swap(self):
        for s, p in zip(self.shadow, self.params):
            tmp = p.detach().clone()
            p.data.copy_(s)
            s.copy_(tmp)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = 'cuda'
    teacher, info = load_teacher(args.teacher, device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    Lt = len(teacher.blocks)
    K = args.K or Lt
    assert Lt % K == 0, 'teacher depth must be a multiple of K'
    stride = Lt // K
    name = f"{info['dataset']}_{args.variant}_K{K}{args.tag}"
    out_dir = os.path.join(HERE, args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    V, Lseq, B = info['vocab'], info['length'], args.batch_size
    student = TiedFMLM(teacher, args.variant, K, args.split, args.groups).to(device)
    ckpt_path = os.path.join(out_dir, 'last.pt')
    if os.path.exists(ckpt_path):
        init = torch.load(ckpt_path, map_location='cpu')['init']
    elif args.init == 'auto':
        gz = torch.Generator(device=device).manual_seed(4321)
        zp = torch.randn((min(B, 64), Lseq, V), device=device, generator=gz)
        before = distill_loss(student, teacher, zp, stride, args.lam)[0].item()
        init, after = search_init(student, teacher, zp, stride, args.lam)
        print(f'init search: {student.init} (loss {before:.4g}) -> {init} (loss {after:.4g})', flush=True)
    elif args.init == 'middle':
        init = student.init
    else:
        init = [int(i) for i in args.init.split(',')]
    if init != student.init:
        student = TiedFMLM(teacher, args.variant, K, args.split, args.groups, init).to(device)
    print('assign', student.assign, 'init', student.init, flush=True)
    params = [p for p in student.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    print(f'{name}: teacher {n_teacher / 1e6:.2f}M params, student {n_student / 1e6:.2f}M '
          f'({n_train / 1e6:.2f}M trainable), compression {n_teacher / n_student:.2f}x', flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    ema = EMA(student.parameters(), args.ema)

    start, log = 0, []
    if os.path.exists(ckpt_path):  # explicit resume
        ck = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ck['student'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        for s, v in zip(ema.shadow, ck['ema']):
            s.copy_(v)
        ema.n = ck.get('ema_n', ck['step'])
        start, log = ck['step'], ck['log']
        print(f'resumed from step {start}', flush=True)

    gen = tok = None
    if not info['is_toy']:
        from probe_onestep import GenPPL
        tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    def save(step):
        torch.save({'student': student.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'ema': ema.shadow, 'ema_n': ema.n, 'init': student.init, 'step': step,
                    'log': log, 'args': vars(args)}, ckpt_path + '.tmp')
        os.replace(ckpt_path + '.tmp', ckpt_path)

    def run_eval(step):
        nonlocal gen
        if not info['is_toy'] and gen is None:
            gen = GenPPL(args.ppl_model, device)
        ema.swap()
        student.eval()
        r = evaluate(student, teacher, info, args, gen, tok)
        student.train()
        ema.swap()
        r['step'] = step
        log.append(r)
        print('[eval]', json.dumps(r), flush=True)
        with open(os.path.join(out_dir, 'eval.json'), 'w') as f:
            json.dump({'name': name, 'args': vars(args), 'init': student.init, 'n_teacher': n_teacher,
                       'n_student': n_student, 'n_trainable': n_train, 'log': log}, f, indent=1)

    t0 = time.time()
    student.train()
    for step in range(start, args.steps):
        z = torch.randn((B, Lseq, V), device=device)
        loss, lh, le = distill_loss(student, teacher, z, stride, args.lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        ema.update()
        if step % args.log_every == 0:
            print(f'step {step} loss {loss.item():.4f} (h {lh.item():.4f}, kl {le.item():.4f}) '
                  f'{time.time() - t0:.0f}s', flush=True)
        if (step + 1) % args.ckpt_every == 0:
            save(step + 1)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            run_eval(step + 1)
    save(args.steps)
    if not log or log[-1]['step'] != args.steps:
        run_eval(args.steps)
    print('done', name, flush=True)


if __name__ == '__main__':
    main()
