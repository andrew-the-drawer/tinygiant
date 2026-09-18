"""Self-speculative decoding: the draft is the same model restricted to
resident experts (zero I/O); the verify pass runs exact routing over all
draft tokens as one batch, so each cold expert is loaded once per round."""

import numpy as np


def dist(logits, temperature, top_p):
    """Sampling distribution over the vocab (one-hot for greedy)."""
    if temperature <= 0:
        p = np.zeros(logits.shape[0], np.float32)
        p[int(np.argmax(logits))] = 1.0
        return p
    z = (logits - logits.max()) / temperature
    p = np.exp(z, dtype=np.float32)
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(p)[::-1]
        cum = np.cumsum(p[order])
        cutoff = int(np.searchsorted(cum, top_p)) + 1
        keep = order[:cutoff]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        q /= q.sum()
        return q
    return p


def sample(p, rng):
    if p.max() >= 1.0:
        return int(np.argmax(p))
    return int(rng.choice(p.shape[0], p=p))


class SpecStats:
    def __init__(self):
        self.rounds = 0
        self.drafted = 0
        self.accepted = 0
        self.emitted = 0

    def as_dict(self):
        return dict(spec_rounds=self.rounds, spec_drafted=self.drafted,
                    spec_accepted=self.accepted,
                    spec_accept_rate=self.accepted / max(1, self.drafted),
                    spec_tokens_per_round=self.emitted / max(1, self.rounds))


def generate_spec(engine, first_logits, start_pos, n_tokens, k, temperature, top_p, rng, stats):
    """Yields tokens. `first_logits` is the exact distribution for position start_pos."""
    pos = start_pos
    t0 = sample(dist(first_logits, temperature, top_p), rng)
    yield t0
    emitted = 1

    while emitted < n_tokens:
        drafts, qs = [], []
        inp = t0
        for i in range(k):
            lg = engine.forward_rows([inp], pos + i, mode="draft")[0]
            q = dist(lg, temperature, top_p)
            d = sample(q, rng)
            drafts.append(d)
            qs.append(q)
            inp = d

        P = engine.forward_rows([t0] + drafts, pos, mode="exact")
        stats.rounds += 1
        stats.drafted += k

        n_acc = 0
        new_tok = None
        for r in range(k):
            p_r = dist(P[r], temperature, top_p)
            d = drafts[r]
            if temperature <= 0:
                accept = int(np.argmax(p_r)) == d
            else:
                accept = rng.random() < min(1.0, p_r[d] / max(qs[r][d], 1e-30))
            if accept:
                n_acc += 1
                emitted += 1
                stats.accepted += 1
                stats.emitted += 1
                yield d
                if emitted >= n_tokens:
                    break
            else:
                if temperature <= 0:
                    new_tok = int(np.argmax(p_r))
                else:
                    res = np.maximum(p_r - qs[r], 0)
                    s = res.sum()
                    res = res / s if s > 0 else p_r
                    new_tok = sample(res, rng)
                break
        if emitted >= n_tokens:
            engine.kv_len = pos + n_acc + 1
            return
        if new_tok is None:
            new_tok = sample(dist(P[k], temperature, top_p), rng)
        emitted += 1
        stats.emitted += 1
        yield new_tok

        engine.kv_len = pos + n_acc + 1
        pos = engine.kv_len
        t0 = new_tok
