"""LLM 客户端：接入火山方舟（Volcengine Ark）OpenAI 兼容接口，用于生成围棋讲解的自然语言文案。"""
import json
import urllib.request
import urllib.error
from typing import Optional

from katrain.core.lang import i18n

DEFAULT_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

# 内置模型（用户只需填 API Key，模型从下拉框选）
BUILTIN_MODELS = [
    ("kimi-k3", "Kimi K3"),
    ("doubao-seed-2.1-turbo", "Doubao Seed 2.1 Turbo"),
    ("deepseek-v4-flash", "DeepSeek V4 Flash"),
]


class LLMError(Exception):
    pass


def is_configured(katrain) -> bool:
    """检查是否配置了可用的 LLM。"""
    return bool(
        katrain.config("llm/api_key", "").strip()
        and katrain.config("llm/model", "").strip()
    )


def get_model_display_name(katrain) -> str:
    model = katrain.config("llm/model", "")
    for mid, mname in BUILTIN_MODELS:
        if mid == model:
            return mname
    return model


def chat_completion(
    katrain,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 2000,
    timeout: int = 60,
) -> str:
    """调用火山方舟 chat completion，返回纯文本。"""
    api_key = katrain.config("llm/api_key", "").strip()
    model = katrain.config("llm/model", "").strip()
    endpoint = katrain.config("llm/endpoint", DEFAULT_ENDPOINT).strip() or DEFAULT_ENDPOINT

    if not api_key:
        raise LLMError(i18n._("LLM API key is not set. Please configure it in General Settings."))
    if not model:
        raise LLMError(i18n._("LLM model is not selected."))

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise LLMError(f"HTTP {e.code}: {body}")
    except Exception as e:
        raise LLMError(str(e))

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"Unexpected LLM response: {json.dumps(data)[:500]}")
