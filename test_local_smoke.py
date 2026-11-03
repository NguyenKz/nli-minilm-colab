"""Quick load smoke (no HF datasets)."""
from model import NLI, collate_fn, tokenizes
from transformers import AutoTokenizer


def main():
    tok = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
    # untrained architecture only — just check shapes
    from transformers import AutoModel

    bb = AutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
    from model import NLIConfig

    bc = bb.config
    cfg = NLIConfig(
        vocab_size=bc.vocab_size,
        hidden_size=bc.hidden_size,
        nclass=3,
        num_hidden_layers=bc.num_hidden_layers,
        num_attention_heads=bc.num_attention_heads,
        intermediate_size=bc.intermediate_size,
        max_position_embeddings=bc.max_position_embeddings,
    )
    m = NLI(cfg)
    m.encoder.load_state_dict(bb.state_dict(), strict=False)
    s = tokenizes("A man runs. [CLS] Someone is moving.", tokenizer=tok)
    out = m(**collate_fn([{"input_ids": s["input_ids"], "labels": 0}]))
    n = sum(p.numel() for p in m.parameters())
    print("params", n, "logits", tuple(out.logits.shape))
    assert n < 40_000_000 and out.logits.shape[-1] == 3
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
