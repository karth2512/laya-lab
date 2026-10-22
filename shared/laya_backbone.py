"""Laya's encoder as a standard sequence classifier (ADR 0001, path A). Used by a2, a3 and a4.

`ModernBertForSequenceClassification` is built from Laya's encoder config and its `.model` gets
the encoder weights of the pinned Laya checkpoint. Arms differ only in what they make trainable.
`backbone.tiny_random: true` swaps in a tiny random model and a character tokenizer (CPU tests).
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from transformers import (
    ModernBertConfig,
    ModernBertForSequenceClassification,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

N_CLASSES = 5
_SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789.,;:()-%/")


def load_classifier(
    cfg: dict[str, Any], device: str, num_labels: int = N_CLASSES
) -> tuple[ModernBertForSequenceClassification, PreTrainedTokenizerBase]:
    """(classifier, tokenizer) on `device`, fp32. Reads `cfg["arm"]["backbone"]`, `cfg["laya"]`."""
    bb = cfg["arm"].get("backbone", {})
    if bb.get("tiny_random", False):
        model, tok = _tiny(num_labels)
    else:
        model, tok = _from_laya(cfg["laya"], num_labels)
    model.config.classifier_pooling = bb.get("classifier_pooling", "mean")
    model.config.classifier_dropout = float(bb.get("classifier_dropout", 0.0))
    model.drop.p = model.config.classifier_dropout
    return model.to(device), tok  # type: ignore[arg-type]


def _from_laya(
    laya_cfg: dict[str, Any], num_labels: int
) -> tuple[ModernBertForSequenceClassification, PreTrainedTokenizerBase]:
    from huggingface_hub import snapshot_download
    from laya.train import load_checkpoint

    model_dir = snapshot_download(
        laya_cfg["hub_id"],
        revision=laya_cfg["revision"],
        allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
    )
    decision_model, tok, _ = load_checkpoint(model_dir)
    encoder = decision_model.encoder
    config = copy.deepcopy(encoder.config)
    config.num_labels = num_labels
    config._attn_implementation = "sdpa"
    model = ModernBertForSequenceClassification(config)
    model.model.load_state_dict(encoder.state_dict(), strict=True)
    del decision_model
    return model.float(), tok


def _tiny(num_labels: int) -> tuple[ModernBertForSequenceClassification, PreTrainedTokenizerBase]:
    from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers, processors

    vocab = {t: i for i, t in enumerate(_SPECIALS + _CHARS)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"\S"), behavior="isolated")
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    config = ModernBertConfig(  # type: ignore[call-arg]
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
        bos_token_id=2,
        eos_token_id=3,
        num_labels=num_labels,
    )
    config._attn_implementation = "sdpa"
    return ModernBertForSequenceClassification(config), tok


def pool(model: Any, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """The classifier's pooling (cls or masked mean), as in ModernBertForSequenceClassification."""
    if model.config.classifier_pooling == "cls":
        return hidden[:, 0]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1)


def head_logits(model: Any, pooled: torch.Tensor) -> torch.Tensor:
    """Head -> dropout -> classifier on pooled features (a2 trains this on cached features)."""
    return model.classifier(model.drop(model.head(pooled)))
