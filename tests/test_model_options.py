"""Unit tests for webapp._model_options's quantization/context-length parsing,
used to auto-fill the run form from a profile's already-pulled models."""

from lemonmatrix.webapp import _model_options


def test_prefers_checkpoint_variant_suffix():
    options = _model_options([{"id": "user.Llama-3.2-1B-Q4_K_M", "checkpoint": "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"}])
    assert options[0]["quantization"] == "Q4_K_M"


def test_falls_back_to_id_suffix_when_no_checkpoint():
    options = _model_options([{"id": "Qwen3.8-27B-GGUF-Q4_K_M"}])
    assert options[0]["quantization"] == "Q4_K_M"


def test_does_not_mistake_gguf_file_format_for_a_quantization():
    # No checkpoint, and the id's only trailing segment is "-GGUF" -- that's
    # the file format, not a quant, and must not be reported as one.
    options = _model_options([{"id": "Llama-3.1-8B-Instruct-GGUF"}])
    assert options[0]["quantization"] == ""


def test_recognizes_common_gguf_quant_conventions():
    for suffix in ["Q8_0", "IQ4_XS", "F16", "BF16", "IQ3_M"]:
        options = _model_options([{"id": f"Model-{suffix}"}])
        assert options[0]["quantization"] == suffix, suffix


def test_context_length_caps_at_default_even_for_huge_max_context_window():
    # Confirmed live: defaulting to a real model's max_context_window
    # (262144) makes even loading painfully slow (huge KV-cache allocation)
    # before any inference runs -- the suggested default must not exceed
    # DEFAULT_BENCH_CONTEXT_LENGTH regardless of what the model supports.
    options = _model_options([{"id": "A", "max_context_window": 262144}])
    assert options[0]["context_length"] == 4096


def test_context_length_respects_a_smaller_max_context_window():
    options = _model_options([{"id": "A", "max_context_window": 2048}])
    assert options[0]["context_length"] == 2048


def test_context_length_falls_back_to_default_when_unknown():
    options = _model_options([{"id": "B"}])
    assert options[0]["context_length"] == 4096
