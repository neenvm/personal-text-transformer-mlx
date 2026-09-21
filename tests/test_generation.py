from pathlib import Path

import mlx.core as mx

from imessage_mlx import generate
from imessage_mlx.generate import format_reply_prompt, generate_ids, stream_reply_with_model
from imessage_mlx.model.config import ModelConfig
from imessage_mlx.model.transformer import TransformerLM
from imessage_mlx.tokenizer.train import load_tokenizer, train_tokenizer
from imessage_mlx.utils import write_jsonl


def test_generation_is_bounded_and_prompt_uses_roles() -> None:
    model = TransformerLM(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            intermediate_size=32,
            max_sequence_length=8,
        )
    )
    model.eval()
    mx.eval(model.parameters())
    generated = generate_ids(
        model,
        [1, 2, 3],
        eos_ids=set(),
        max_new_tokens=5,
        temperature=0.0,
    )
    assert len(generated) == 5
    prompt = format_reply_prompt("hello", [("other", "earlier"), ("me", "reply")])
    assert prompt.startswith("<|bos|><|conversation|>")
    assert prompt.endswith("<|me|>")
    assert "<|other|>hello<|turn_end|>" in prompt


def test_streaming_preserves_multibyte_text(monkeypatch, tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    write_jsonl(train, [{"text": "😀 💙 café naïve hello world"}])
    train_tokenizer(train, tmp_path / "tokenizer", vocab_size=256, minimum_frequency=2)
    tokenizer = load_tokenizer(tmp_path / "tokenizer")
    expected = "😀 café"
    generated_ids = tokenizer.encode(expected, add_special_tokens=False).ids

    def fake_generated_ids(*args, **kwargs):
        yield from generated_ids

    monkeypatch.setattr(generate, "iter_generated_ids", fake_generated_ids)
    chunks = list(stream_reply_with_model(object(), tokenizer, "hello"))

    assert "".join(chunks) == expected
    assert "�" not in "".join(chunks)
