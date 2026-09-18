"""Expert-routing predictors used to prefetch cold experts ahead of demand."""

import numpy as np


class LookaheadPredictor:
    """Applies layer (l+j)'s router to the residual stream at layer l."""

    def __init__(self, routers, ffn_norms, eps, biases=None):
        self.routers = routers
        self.ffn_norms = ffn_norms
        self.eps = eps
        self.n_layers = len(routers)
        # Fold the RMS-norm weight into each router so lookahead over several
        # target layers is a single matmul against the normalized residual.
        self.folded = np.ascontiguousarray(
            np.stack([r * n[None, :] for r, n in zip(routers, ffn_norms)]), dtype=np.float32)
        self.n_experts = self.folded.shape[1]
        self.biases = np.stack(biases).astype(np.float32) if biases is not None else None

    def candidates(self, l0, l1, X, n):
        """Boolean [l1-l0, n_experts] of the top-n predicted experts for target
        layers l0..l1-1 (union over rows of X [k, D])."""
        rr = np.sqrt(np.mean(X * X, axis=1, keepdims=True) + self.eps)
        Xn = X / rr
        J = l1 - l0
        logits = (Xn @ self.folded[l0:l1].reshape(J * self.n_experts, -1).T).reshape(-1, J, self.n_experts)
        if self.biases is not None:
            logits = logits + self.biases[l0:l1][None]
        idx = np.argpartition(logits, -n, axis=2)[:, :, -n:]     # [k, J, n]
        cand = np.zeros((J, self.n_experts), bool)
        jj = np.broadcast_to(np.arange(J)[None, :, None], idx.shape)
        cand[jj.ravel(), idx.ravel()] = True
        return cand

    def predict(self, target_layer, x, n):
        rr = np.sqrt(np.mean(x * x) + self.eps)
        normed = (x / rr) * self.ffn_norms[target_layer]
        logits = self.routers[target_layer] @ normed
        if self.biases is not None:
            logits = logits + self.biases[target_layer]
        idx = np.argpartition(logits, -n)[-n:]
        return idx[np.argsort(logits[idx])[::-1]]


class TokenPrior:
    """Per-layer table: token id -> experts most often routed to, from traces."""

    def __init__(self, table, min_count=2):
        self.table = table          # {layer: {token: (experts[int], counts[int])}}
        self.min_count = min_count

    @classmethod
    def from_traces(cls, paths, n_experts, k_keep=8, min_count=2):
        counts = {}
        for p in paths:
            d = np.load(p)
            tokens, selected = d["tokens"], d["selected"]
            T, L, K = selected.shape
            for t in range(T):
                tok = int(tokens[t])
                for l in range(L):
                    c = counts.setdefault(l, {}).setdefault(tok, np.zeros(n_experts, np.int32))
                    c[selected[t, l]] += 1
        table = {}
        for l, per_tok in counts.items():
            table[l] = {}
            for tok, c in per_tok.items():
                order = np.argsort(c)[::-1][:k_keep]
                order = order[c[order] > 0]
                table[l][tok] = (order.astype(np.int16), c[order])
        return cls(table, min_count)

    def predict(self, layer, token_id, n):
        ent = self.table.get(layer, {}).get(int(token_id))
        if ent is None:
            return np.zeros(0, np.int16)
        experts, counts = ent
        return experts[:n][counts[:n] >= self.min_count]

    def save(self, path):
        flat = {}
        for l, per_tok in self.table.items():
            for tok, (ex, ct) in per_tok.items():
                flat[f"{l}_{tok}_e"] = ex
                flat[f"{l}_{tok}_c"] = ct
        np.savez_compressed(path, **flat)

    @classmethod
    def load(cls, path, min_count=2):
        d = np.load(path)
        table = {}
        for key in d.files:
            if not key.endswith("_e"):
                continue
            l, tok, _ = key.split("_")
            table.setdefault(int(l), {})[int(tok)] = (d[key], d[f"{l}_{tok}_c"])
        return cls(table, min_count)
