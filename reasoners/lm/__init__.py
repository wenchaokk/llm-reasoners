# Lazy imports so Colab/lightweight envs don't need fairscale until LlamaModel is used
__all__ = [
    "HFModel",
    "LlamaModel",
    "Llama2Model",
    "Llama3Model",
    "LlamaCppModel",
    "OpenAIModel",
    "OpenAIModelWithUsage",
    "ExLlamaModel",
    "BardCompletionModel",
    "ClaudeModel",
    "SGLangModel",
]

_MODULES = {
    "HFModel": ".hf_model",
    "LlamaModel": ".llama_model",
    "Llama2Model": ".llama_2_model",
    "Llama3Model": ".llama_3_model",
    "LlamaCppModel": ".llama_cpp_model",
    "OpenAIModel": ".openai_model",
    "OpenAIModelWithUsage": ".openai_model_with_usage",
    "ExLlamaModel": ".exllama_model",
    "BardCompletionModel": ".gemini_model",
    "ClaudeModel": ".anthropic_model",
    "SGLangModel": ".sglang_model",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(_MODULES[name], package=__name__)
    return getattr(mod, name)
