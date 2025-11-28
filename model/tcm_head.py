import torch
import torch.nn as nn
import torch.nn.functional as F


def build_tcm_vocab(converter, add_tokens=("<pad>",)):
    base = list(converter.character)
    stoi = {ch: i for i, ch in enumerate(base)}
    for t in add_tokens:
        if t not in stoi:
            stoi[t] = len(stoi)
    itos = [''] * len(stoi)
    for k, v in stoi.items():
        itos[v] = k
    pad_id = stoi["<pad>"]
    return stoi, itos, pad_id


def texts_to_ids(texts, stoi):
    return [torch.tensor([stoi[ch] for ch in t], dtype=torch.long) for t in texts]


def make_context_batch(texts, stoi, sub_str_len=5, device='cuda'):
    ids = [torch.tensor([stoi[ch] for ch in t], dtype=torch.long, device=device) for t in texts]
    B = len(ids); Lmax = max(t.size(0) for t in ids); S = sub_str_len
    PAD = stoi["<pad>"]

    left  = torch.full((B, Lmax, S), PAD, dtype=torch.long, device=device)
    right = torch.full((B, Lmax, S), PAD, dtype=torch.long, device=device)
    tgt   = torch.full((B, Lmax),    PAD, dtype=torch.long, device=device)
    mask  = torch.zeros((B, Lmax),   dtype=torch.float32,   device=device)

    for b, seq in enumerate(ids):
        L = seq.size(0)
        tgt[b, :L]  = seq
        mask[b, :L] = 1.0
        for i in range(L):
            l_ctx = seq[max(0, i-S):i]
            # left pad with PAD
            if l_ctx.numel() < S:
                l_ctx = torch.cat([torch.full((S - l_ctx.numel(),), PAD, device=device), l_ctx], dim=0)
            left[b, i] = l_ctx[-S:]

            r_ctx = seq[i+1:min(L, i+1+S)]
            # right pad with PAD
            if r_ctx.numel() < S:
                r_ctx = torch.cat([r_ctx, torch.full((S - r_ctx.numel(),), PAD, device=device)], dim=0)
            right[b, i] = r_ctx[:S]

    return left, right, tgt, mask


class TCMHead(nn.Module):
    def __init__(self, d_vis, vocab_size_tcm, pad_id, d_txt=256, sub_str_len=5, p_drop=0.1):
        super().__init__()
        self.vocab_size = vocab_size_tcm
        self.sub_str_len = sub_str_len

        # critical: padding_idx zeroes the PAD row and keeps it frozen
        self.emb = nn.Embedding(vocab_size_tcm, d_txt, padding_idx=pad_id)

        # keep direction as learned vectors (not tokens)
        self.dir_left  = nn.Parameter(torch.randn(1, 1, d_txt))
        self.dir_right = nn.Parameter(torch.randn(1, 1, d_txt))

        self.ctx_conv = nn.Conv1d(d_txt, d_txt, kernel_size=3, padding=1)
        self.txt_proj = nn.Linear(d_txt, d_vis)
        self.q_norm   = nn.LayerNorm(d_vis)
        self.kv_norm  = nn.LayerNorm(d_vis)
        self.dropout  = nn.Dropout(p_drop)
        self.classifier = nn.Linear(d_vis, vocab_size_tcm)


    def _context_to_query(self, ctx_ids, dir_token):
        E = self.emb(ctx_ids)
        B, L, S, D = E.shape
        x = E.view(B*L, S, D).transpose(1, 2)
        x = self.ctx_conv(x)
        x = x.mean(dim=-1)
        x = x.view(B, L, D)

        x = x + dir_token
        x = self.txt_proj(x)
        return self.q_norm(x)

    def _cross_attend(self, Q, F):
        K = self.kv_norm(F)
        V = K
        attn = torch.einsum('bld,bnd->bln', Q, K) / \
            (K.size(-1) ** 0.5)
        A = attn.softmax(dim=-1)
        out = torch.einsum('bln,bnd->bld', A, V)
        return self.dropout(out)

    def forward(self,
                vis_tokens,
                left_ctx_ids,
                right_ctx_ids,
                tgt_ids,
                tgt_mask,
                focus_mask=None):
        Ql = self._context_to_query(left_ctx_ids,  self.dir_left)
        Qr = self._context_to_query(right_ctx_ids, self.dir_right)

        Fl = self._cross_attend(Ql, vis_tokens)
        Fr = self._cross_attend(Qr, vis_tokens)

        logits_l = self.classifier(Fl)
        logits_r = self.classifier(Fr)

        loss_l = F.cross_entropy(
            logits_l.view(-1, self.vocab_size),
            tgt_ids.view(-1),
            reduction='none'
        ).view_as(tgt_ids)
        loss_r = F.cross_entropy(
            logits_r.view(-1, self.vocab_size),
            tgt_ids.view(-1),
            reduction='none'
        ).view_as(tgt_ids)

        if focus_mask is not None:
            weights = tgt_mask * (1.0 + focus_mask)
        else:
            weights = tgt_mask

        loss_masked = (loss_l + loss_r) * weights
        denom = torch.clamp(weights.sum(), min=1.0)
        loss_tcm = loss_masked.sum() / (2.0 * denom)

        return {'loss_tcm': loss_tcm,
                'logits_l': logits_l,
                'logits_r': logits_r}
