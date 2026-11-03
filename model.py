"""NLI wrapper for CodaBench grader (test.py contract).

Grader builds each sample as:  premise + " [CLS] " + hypothesis
We re-parse that string into a real sentence-pair so the encoder sees
[CLS] premise [SEP] hypothesis [SEP] with segment ids — not a flat bag of tokens.
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

# WordPiece specials shared by BERT / MiniLM uncased vocabs
_PAD, _CLS, _SEP = 0, 101, 102
_MAX_LEN = 128
# grader's join marker (must match test.py exactly)
_JOIN = " [CLS] "


def _premise_hypothesis(text: str):
    """Recover (premise, hypothesis) from grader's concatenated string."""
    cut = text.find(_JOIN)
    if cut < 0:
        # fallback: treat whole string as premise if marker missing
        return text, ""
    return text[:cut], text[cut + len(_JOIN) :]


def tokenizes(examples, tokenizer):
    """Encode one sample (str) the way test.py calls us: tokenizes(sample, tokenizer=...)."""
    prem, hypo = _premise_hypothesis(examples)
    enc = tokenizer(
        prem,
        hypo,
        truncation=True,
        max_length=_MAX_LEN,
        # dynamic pad happens in collate_fn (grader only keeps input_ids)
        padding=False,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    return enc


def collate_fn(batch):
    """Stack variable-length pair encodings; rebuild mask + token_type_ids."""
    # drop pad leftovers if any, then right-pad to longest in the minibatch
    seqs = []
    for row in batch:
        ids = [int(t) for t in row["input_ids"] if int(t) != _PAD]
        seqs.append(ids[:_MAX_LEN])

    width = max(len(s) for s in seqs)
    bsz = len(seqs)
    input_ids = torch.full((bsz, width), _PAD, dtype=torch.long)
    token_type_ids = torch.zeros((bsz, width), dtype=torch.long)

    for i, ids in enumerate(seqs):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        # tokens after the first [SEP] belong to the hypothesis segment
        try:
            sep_at = ids.index(_SEP)
            token_type_ids[i, sep_at + 1 : n] = 1
        except ValueError:
            pass

    out = {
        "input_ids": input_ids,
        "attention_mask": (input_ids != _PAD).long(),
        "token_type_ids": token_type_ids,
    }
    if "labels" in batch[0]:
        out["labels"] = torch.tensor([int(x["labels"]) for x in batch])
    return out


class NLIConfig(PretrainedConfig):
    model_type = "NLI"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=384,
        nclass=3,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=1536,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.nclass = nclass
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob


class NLI(PreTrainedModel):
    """MiniLM-sized BERT encoder + small MLP head on the [CLS] vector."""

    config_class = NLIConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        enc_cfg = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        )
        self.encoder = BertModel(enc_cfg)
        # light 2-layer head (different from a single Linear classifier)
        h = config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(h, config.nclass),
        )
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing=0.03)
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = enc.last_hidden_state[:, 0]  # [CLS] pooled
        logits = self.head(cls)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
