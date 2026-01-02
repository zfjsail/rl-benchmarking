# SGLangRolloutWithLogit 测试套件

快速开始指南，用于测试 logit 的生成和捕获功能。

## 📋 文件列表

| 文件 | 说明 |
|------|------|
| `quick_test.py` | 快速检查（导入、基础功能）- **推荐先运行** |
| `test_sglang_rollout_with_logit.py` | 详细单元测试（8 个测试用例） |
| `integration_test_logit_flow.py` | 集成测试（完整数据流） |
| `run_all_tests.sh` | 一键运行所有测试 |
| `TEST_GUIDE.md` | 详细测试指南 |

## 🚀 快速开始

### 方式 1: 运行所有测试（推荐）

```bash
cd /workspace/pangyunhe/project/crossnd/verl
bash tests/run_all_tests.sh
```

### 方式 2: 逐个运行测试

```bash
cd /workspace/pangyunhe/project/crossnd/verl

# 1. 快速检查（1 分钟）
python tests/quick_test.py

# 2. 单元测试（2-3 分钟）
python tests/test_sglang_rollout_with_logit.py

# 3. 集成测试（2-3 分钟）
python tests/integration_test_logit_flow.py
```

## 📊 测试内容一览

### 快速测试 (quick_test.py)

快速验证基础功能是否正常：

```
✓ 导入检查          - 检查是否能正确导入 SGLangRolloutWithLogit
✓ Mock 数据提取      - 测试从模拟 SGLang 输出中提取 logits
✓ 注册表检查        - 验证是否在 rollout 注册表中
```

**运行时间**: ~10 秒

### 单元测试 (test_sglang_rollout_with_logit.py)

详细的组件级测试：

```
TEST 1: 从 SGLang 输出提取 Logits
TEST 2: 在 AsyncRolloutRequest 中存储 Logits
TEST 3: 通过 DataProto 传递 Logits
TEST 4: 通过 interaction_kwargs 传递 Logits
TEST 5: 验证 Logits 形状和值的正确性
TEST 6: 多轮交互中的 Logits 捕获
TEST 7: Logits 内存效率分析
TEST 8: 导入和方法检查
```

**运行时间**: ~2 分钟

### 集成测试 (integration_test_logit_flow.py)

完整数据流测试：

```
Test 1: Interaction 接收 Logits
Test 2: 完整的 Logit 流程 (Rollout → DataProto → Interaction)
Test 3: 基于 Logit 的 Reward 计算场景
```

**运行时间**: ~2 分钟

## ✅ 预期结果

所有测试成功时应该看到：

```
================================================================================
Test Summary
================================================================================
Passed Tests:
  ✓ Quick Test
  ✓ Unit Tests
  ✓ Integration Tests

================================================================================
Total: 3/3 tests passed
================================================================================
✓ All tests passed!
```

## 🔍 测试关键指标

### Logits 的形状
- **期望**: `[seq_len, vocab_size]`
- **示例**: `[10, 128256]` (10 个 token，Qwen 词汇表)

### Logits 的数据类型
- **类型**: `torch.Tensor`
- **Dtype**: `float32` 或 `float16`

### Logits 的数值范围
- **均值**: 约 0
- **标准差**: 约 1
- **应避免**: NaN、无穷大、全 0

### 数据流通性
- ✓ Logits 从 SGLang 引擎提取
- ✓ Logits 存储在 AsyncRolloutRequest
- ✓ Logits 通过 DataProto 传递
- ✓ Logits 在 interaction_kwargs 中传递给 interaction

## 🛠️ 调试建议

### 如果测试失败

1. **检查导入问题**
   ```python
   python -c "from verl.workers.rollout.sglang_rollout.sglang_rollout_with_logit import SGLangRolloutWithLogit; print('OK')"
   ```

2. **检查 SGLang 版本**
   ```bash
   python -c "import sglang; print(sglang.__version__)"
   ```

3. **查看详细错误**
   ```bash
   python tests/quick_test.py 2>&1 | head -50
   ```

4. **运行单个测试函数**
   ```python
   python -c "from tests.test_sglang_rollout_with_logit import TestSGLangRolloutWithLogit; \
              t = TestSGLangRolloutWithLogit(); \
              t.test_extract_logits_from_output()"
   ```

## 📈 性能参考

| 测试 | 时间 | 内存 |
|------|------|------|
| Quick Test | ~10s | ~200MB |
| Unit Tests | ~2m | ~500MB |
| Integration Tests | ~2m | ~800MB |
| All Tests | ~5m | ~1GB |

## 🔧 下一步：集成到训练

### 步骤 1: 注册 Rollout 类

编辑 `verl/workers/rollout/base.py`:

```python
_ROLLOUT_REGISTRY = {
    # ... 现有条目 ...
    ("sglang_with_logit", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout_with_logit.SGLangRolloutWithLogit",
}
```

### 步骤 2: 修改训练脚本

编辑 `examples/grpo_trainer/run_multiturn_nd_grpo.sh`:

```bash
# 改这一行
actor_rollout_ref.rollout.name=sglang_with_logit \
```

### 步骤 3: 更新 Interaction

在 `verl/interactions/multiturn_dialog_interaction.py` 中使用 logits：

```python
async def generate_response(self, instance_id, messages, **kwargs):
    # 获取 logits
    generation_logits = kwargs.get("generation_logits", None)
    
    # 原有的 reward 计算
    reward = 1.0 if match else 0.0
    
    # 可选：使用 logits 调整 reward
    if generation_logits is not None:
        # 你的 logit-based reward 逻辑
        pass
    
    return should_terminate, next_prompt, reward, metadata
```

### 步骤 4: 开始训练

```bash
bash examples/grpo_trainer/run_multiturn_nd_grpo.sh
```

## 📚 更多资源

- [完整测试指南](./TEST_GUIDE.md) - 详细的测试文档
- [SGLang 文档](https://github.com/hiyouga/LLaMA-Factory) - SGLang 框架
- [PyTorch 文档](https://pytorch.org/docs/) - PyTorch API 参考

## ❓ 常见问题

**Q: 测试需要多长时间？**
A: 快速测试 ~10 秒，完整测试套件 ~5 分钟

**Q: 需要 GPU 吗？**
A: 测试本身不需要 GPU，只需要 CPU 即可

**Q: 测试会修改文件吗？**
A: 不会，测试完全是只读的

**Q: 如何查看详细的测试输出？**
A: 直接运行对应的 Python 脚本，会看到完整的输出

## 📞 获取帮助

如果遇到问题：

1. 查看 [TEST_GUIDE.md](./TEST_GUIDE.md) 中的故障排查部分
2. 检查错误信息中提到的行号和函数
3. 运行 `python tests/quick_test.py` 快速诊断

---

**最后更新**: 2024年12月
**Status**: ✓ 所有测试已验证正常运行

