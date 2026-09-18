import os


class _LlamaTok:
    def __init__(self, model_path):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=32, n_gpu_layers=0,
                         vocab_only=True, verbose=False)
        self.eos_id = self.llm.token_eos()

    def encode(self, text):
        return self.llm.tokenize(text.encode(), add_bos=True)

    def decode(self, ids):
        return self.llm.detokenize(ids).decode("utf-8", errors="replace")


class _HFTok:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(path)
        self.eos_id = self.tok.token_to_id("<|im_end|>")

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)


def load_tokenizer(model_path):
    """Prefer llama-cpp-python; fall back to a tokenizer.json next to the model."""
    try:
        return _LlamaTok(model_path)
    except ImportError:
        pass
    candidates = [
        os.path.join(os.path.dirname(model_path), "tokenizer.json"),
        os.path.expanduser("~/models/tokenizer.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return _HFTok(p)
    raise RuntimeError("No tokenizer: install llama-cpp-python or place tokenizer.json next to the model")
