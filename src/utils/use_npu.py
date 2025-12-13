import os
import torch
from contextlib import contextmanager
from .tag2ansi import tag2ansi


## 存在 USE_NPU 环境变量时，将代码移动到华为昇腾 NPU 上运行
USE_NPU = os.environ.get('USE_NPU', None) not in [None, '', 'no', 'No', 'false', 'False']
if USE_NPU:
    print(tag2ansi("[yellow bold]Using Huawei Ascend NPU for training.[reset]"))
    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # 无感迁移到 NPU
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def npu_attention_fallback(model):
    """
    将模型中的 MultiheadAttention 强制切换到 train 模式并关闭 dropout，
    以避开 NPU 上不支持的 _native_multi_head_attention 融合算子。

    Args:
        model: PyTorch 模型
    """
    original_states = {}
    # 保存原始的 dropout 和 training 状态
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            original_states[name] = {
                'dropout': module.dropout,
                'training': module.training
            }
            module.train()  # 强制开启 train 模式 (fallback 到 MatMul 路径)
            module.dropout = 0.0  # 强制关闭 dropout (保证确定性)
            # 不需要考虑 BN 等因素，因为 MultiheadAttention 里没有
    return original_states


def recover_npu_attention(model, original_states):
    """
    恢复模型中 MultiheadAttention 的 dropout 和 training 状态。

    Args:
        model: PyTorch 模型
        original_states: 通过 npu_attention_fallback 保存的原始状态字典
    """
    for name, module in model.named_modules():
        if name in original_states:
            state = original_states[name]
            module.dropout = state['dropout'] # 恢复 dropout
            module.train(state['training']) # 恢复原本的模式 (train 或 eval)


@contextmanager
def npu_attention_fallback_context(model, enable=True):
    """
    上下文管理器：临时将 MultiheadAttention 切换到 train 模式并关闭 dropout，
    以避开 NPU 上不支持的 _native_multi_head_attention 融合算子。
    
    Args:
        model: PyTorch 模型
        enable: 是否开启此 workaround (方便通过 args 控制)
    """
    if not enable:
        yield
        return
    else:
        original_states = npu_attention_fallback(model)
        try:
            yield
        finally:
            recover_npu_attention(model, original_states)
