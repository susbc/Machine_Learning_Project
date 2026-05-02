"""
tests/test_llm_tuning.py — LLM 调参鲁棒解析单元测试。

不写 LLM 调用本身的集成测试（依赖外部 API key）。
只测试 _extract_json, _clip_and_coerce 的纯函数行为。
"""

import pytest
from web_app.manual_training.llm_tuning import LLMTuningError, _extract_json, _clip_and_coerce


def test_parse_clean_json_block():
    """完美 ```json``` 块 → 解析成功。"""
    text = """
Here is my analysis...

```json
{
  "params": {"n_estimators": 300, "learning_rate": 0.05},
  "reasoning": "Based on the data characteristics..."
}
```

Hope this helps!
"""
    result = _extract_json(text)
    assert '"n_estimators"' in result
    assert '"learning_rate"' in result


def test_parse_dirty_response():
    """LLM 在 JSON 前后有解释文本 → 仍能提取。"""
    text = """
好的，让我来分析一下。基于数据特征，我推荐以下参数：

首先，learning_rate 应该降低到 0.05 左右。
其次，n_estimators 可以增加到 300。

```json
{
  "params": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 8},
  "reasoning": "降低学习率增加树数量是常见的调参策略"
}
```

以上就是我的建议，你可以根据实际情况微调。
"""
    result = _extract_json(text)
    assert '"n_estimators": 300' in result


def test_parse_no_json():
    """全是文本无 JSON → 抛 LLMTuningError。"""
    text = """
我认为应该把学习率调到 0.01，树的数量 200 就够了。
max_depth 设为 6 比较合适。
"""
    with pytest.raises(LLMTuningError, match="非 JSON"):
        _extract_json(text)


def test_clip_out_of_range():
    """learning_rate=2.0 → 截到 0.3。"""
    params = {"learning_rate": 2.0, "n_estimators": 200}
    clipped, warnings = _clip_and_coerce(params, "lightgbm")
    assert clipped["learning_rate"] == 0.3
    assert any("超出上限" in w for w in warnings)


def test_filter_unknown_params():
    """返回 gamma=0.1 但 LightGBM 不暴露 → 过滤。"""
    params = {"gamma": 0.1, "n_estimators": 200}
    clipped, warnings = _clip_and_coerce(params, "lightgbm")
    assert "gamma" not in clipped
    assert "n_estimators" in clipped


def test_string_value_coercion():
    """"300" → int(300)。"""
    params = {"n_estimators": "300", "learning_rate": "0.05"}
    clipped, warnings = _clip_and_coerce(params, "lightgbm")
    assert clipped["n_estimators"] == 300
    assert isinstance(clipped["n_estimators"], int)
    assert clipped["learning_rate"] == 0.05
    assert isinstance(clipped["learning_rate"], float)
