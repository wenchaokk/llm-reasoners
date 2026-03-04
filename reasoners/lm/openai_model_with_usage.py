"""
OpenAI 模型 + Token/费用实时监控

在 OpenAIModel 基础上增加每次调用的 Token 统计和费用估算。
每次 generate 调用后会打印本次及累计使用量。
"""

from typing import Optional

from reasoners.lm.openai_model import OpenAIModel

# OpenAI 定价（美元/百万 tokens，2024 年参考价，请以官网为准）
# 来源：https://openai.com/api/pricing/
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


def _get_model_pricing(model_name: str) -> tuple[float, float]:
    """根据模型名获取输入/输出单价（美元/百万 tokens）"""
    model_lower = model_name.lower()
    for key in PRICING:
        if key in model_lower:
            return PRICING[key]["input"], PRICING[key]["output"]
    return PRICING["gpt-4o-mini"]["input"], PRICING["gpt-4o-mini"]["output"]


class OpenAIModelWithUsage(OpenAIModel):
    """带 Token 和费用监控的 OpenAI 模型

    每次 generate 调用后会实时打印：
    - 本次：prompt_tokens, completion_tokens, total_tokens, 本次费用
    - 累计：总 tokens, 总费用
    """

    def __init__(self, *args, **kwargs):
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost_usd = 0.0
        self._call_count = 0
        kwargs["usage_callback"] = self._on_usage
        super().__init__(*args, **kwargs)

    def _on_usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        """每次 API 调用后的回调：记录并打印"""
        input_price, output_price = _get_model_pricing(self.model)
        cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_cost_usd += cost
        self._call_count += 1

        print(
            f"[Token 监控] 本次: prompt={prompt_tokens}, completion={completion_tokens}, "
            f"total={total_tokens} | 费用≈${cost:.6f}"
        )
        print(
            f"[Token 监控] 累计: total={self._total_prompt_tokens + self._total_completion_tokens} tokens, "
            f"费用≈${self._total_cost_usd:.4f} (共 {self._call_count} 次调用)"
        )

    def get_usage_summary(self) -> dict:
        """获取使用统计摘要"""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "cost_usd": self._total_cost_usd,
            "call_count": self._call_count,
        }

    def reset_usage(self):
        """重置累计统计"""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost_usd = 0.0
        self._call_count = 0
