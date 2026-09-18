"""Top-k expert routing with cache-aware selection modes.

Selection can be biased toward resident experts; gate weights are always
computed from the unbiased logits (the bias only changes *which* experts run).
"""

import numpy as np


class RoutingConfig:
    def __init__(self, bias=0.0, margin=None, tau=None, protect_top=0, skip_prob=0.0):
        self.bias = bias
        self.margin = margin
        self.tau = tau
        self.protect_top = protect_top
        self.skip_prob = skip_prob


class Router:

    def __init__(self, routers, n_used):
        self.routers = routers
        self.k = n_used
        self.n_experts = routers[0].shape[0]
        self.stats = dict(slots=0, swapped=0, displaced_mass=0.0, restricted_fallback=0)
        self.rng = np.random.default_rng(0)

    def logits(self, layer, normed):
        return self.routers[layer] @ normed

    def _topk(self, v):
        idx = np.argpartition(v, -self.k)[-self.k:]
        return idx[np.argsort(v[idx])[::-1]]

    def _weights(self, logits, idx):
        l = logits[idx]
        e = np.exp(l - l.max())
        return (e / e.sum()).astype(np.float32)

    def select(self, logits, resident=None, cfg=None, restrict=False):
        """Returns (idx, weights). idx sorted by descending unbiased logit."""
        k = self.k
        if restrict:
            if resident is None or resident.sum() < k:
                self.stats["restricted_fallback"] += 1
                idx = self._topk(logits)
            else:
                masked = np.where(resident, logits, -np.inf)
                idx = self._topk(masked)
            return idx, self._weights(logits, idx)

        unb = self._topk(logits)
        if cfg is None or cfg.bias == 0.0 or resident is None:
            return unb, self._weights(logits, unb)
        if cfg.skip_prob > 0 and self.rng.random() < cfg.skip_prob:
            return unb, self._weights(logits, unb)

        biased = self._topk(logits + cfg.bias * resident)
        self.stats["slots"] += k
        unb_set = set(unb.tolist())
        b_set = set(biased.tolist())
        if b_set == unb_set:
            return unb, self._weights(logits, unb)

        removed = [e for e in unb if e not in b_set]        # descending logit
        added = [e for e in biased if e not in unb_set]     # descending logit
        probs = None
        if cfg.tau is not None:
            ex = np.exp(logits - logits.max())
            probs = ex / ex.sum()
        protected = set(unb[:cfg.protect_top].tolist()) if cfg.protect_top else set()

        keep = list(unb)
        for rem, add in zip(reversed(removed), added):
            if rem in protected:
                continue
            if cfg.margin is not None and logits[rem] - logits[add] > cfg.margin:
                continue
            if probs is not None and probs[rem] > cfg.tau:
                continue
            keep[keep.index(rem)] = add
            self.stats["swapped"] += 1
            if probs is not None:
                self.stats["displaced_mass"] += float(probs[rem] - probs[add])
        idx = np.array(keep)
        idx = idx[np.argsort(logits[idx])[::-1]]
        return idx, self._weights(logits, idx)
