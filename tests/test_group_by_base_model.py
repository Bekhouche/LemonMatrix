from lemonmatrix.webapp import _group_by_base_model, _model_options


def test_groups_two_quants_of_the_same_base_model_together():
    models = [
        {"id": "user.Llama-3.2-1B-Q4_K_M", "checkpoint": "bartowski/Llama-3.2-1B-GGUF:Q4_K_M"},
        {"id": "user.Llama-3.2-1B-Q8_0", "checkpoint": "bartowski/Llama-3.2-1B-GGUF:Q8_0"},
    ]
    groups = _group_by_base_model(_model_options(models))
    assert list(groups.keys()) == ["user.Llama-3.2-1B"]
    assert {v["quantization"] for v in groups["user.Llama-3.2-1B"]} == {"Q4_K_M", "Q8_0"}


def test_unrelated_models_stay_in_separate_groups():
    models = [{"id": "Llama-3.1-8B-Instruct-GGUF"}, {"id": "Qwen3.8-27B-GGUF-Q4_K_M", "checkpoint": "unsloth/Qwen3.8-27B-GGUF:Q4_K_M"}]
    groups = _group_by_base_model(_model_options(models))
    assert set(groups.keys()) == {"Llama-3.1-8B-Instruct-GGUF", "Qwen3.8-27B-GGUF"}
    assert len(groups["Llama-3.1-8B-Instruct-GGUF"]) == 1
    assert len(groups["Qwen3.8-27B-GGUF"]) == 1


def test_preserves_first_seen_order():
    models = [{"id": "B-Q4_K_M", "checkpoint": "x/B:Q4_K_M"}, {"id": "A-Q4_K_M", "checkpoint": "x/A:Q4_K_M"}]
    groups = _group_by_base_model(_model_options(models))
    assert list(groups.keys()) == ["B", "A"]
