"""Records per-token, per-layer routing state for offline analysis."""

import numpy as np


class TraceRecorder:

    def __init__(self, n_layers, n_experts, k, embed_dim, keep_hidden=True):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.k = k
        self.embed_dim = embed_dim
        self.keep_hidden = keep_hidden
        self.tokens = []
        self.positions = []
        self.logits = []      # [T][L, E]
        self.selected = []    # [T][L, k]
        self.resident = []    # [T][L, E] bool
        self.hidden_in = []   # [T][L, D] f16: residual entering the layer
        self._cur = None

    def begin_token(self, token_id, pos):
        self._cur = dict(
            logits=np.zeros((self.n_layers, self.n_experts), np.float32),
            selected=np.zeros((self.n_layers, self.k), np.int16),
            resident=np.zeros((self.n_layers, self.n_experts), bool),
            hidden_in=np.zeros((self.n_layers, self.embed_dim), np.float16) if self.keep_hidden else None,
        )
        self.tokens.append(token_id)
        self.positions.append(pos)

    def record_layer(self, layer, logits, selected, resident, hidden_in):
        c = self._cur
        c["logits"][layer] = logits
        c["selected"][layer, :len(selected)] = selected
        if resident is not None:
            c["resident"][layer] = resident
        if self.keep_hidden and hidden_in is not None:
            c["hidden_in"][layer] = hidden_in

    def end_token(self):
        c = self._cur
        self.logits.append(c["logits"])
        self.selected.append(c["selected"])
        self.resident.append(c["resident"])
        if self.keep_hidden:
            self.hidden_in.append(c["hidden_in"])
        self._cur = None

    def save(self, path):
        data = dict(
            tokens=np.array(self.tokens, np.int32),
            positions=np.array(self.positions, np.int32),
            logits=np.stack(self.logits),
            selected=np.stack(self.selected),
            resident=np.stack(self.resident),
        )
        if self.keep_hidden:
            data["hidden_in"] = np.stack(self.hidden_in)
        np.savez_compressed(path, **data)
        return path
