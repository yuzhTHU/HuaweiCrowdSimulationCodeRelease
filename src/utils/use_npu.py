import os
import torch
from contextlib import contextmanager
from .tag2ansi import tag2ansi


## When the USE_NPU environment variable is present, run the code on a Huawei Ascend NPU.
USE_NPU = os.environ.get('USE_NPU', None) not in [None, '', 'no', 'No', 'false', 'False']
if USE_NPU:
    print(tag2ansi("[yellow bold]Using Huawei Ascend NPU for training.[reset]"))
    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # Transparent migration to NPU.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def npu_attention_fallback(model):
    """
    Force every `MultiheadAttention` module into train mode and disable dropout
    to avoid the unsupported `_native_multi_head_attention` fused operator on NPU.

    Args:
        model: PyTorch model.
    """
    original_states = {}
    # Preserve the original dropout and training states.
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            original_states[name] = {
                'dropout': module.dropout,
                'training': module.training
            }
            module.train()  # Force train mode to fall back to the MatMul path.
            module.dropout = 0.0  # Force dropout off to keep inference deterministic.
            # No need to consider BatchNorm-like modules here; MultiheadAttention has none.
    return original_states


def recover_npu_attention(model, original_states):
    """
    Restore the original dropout and training states of `MultiheadAttention`.

    Args:
        model: PyTorch model.
        original_states: State dictionary captured by `npu_attention_fallback`.
    """
    for name, module in model.named_modules():
        if name in original_states:
            state = original_states[name]
            module.dropout = state['dropout'] # Restore dropout.
            module.train(state['training']) # Restore the original mode (train or eval).


@contextmanager
def npu_attention_fallback_context(model, enable=True):
    """
    Context manager that temporarily switches `MultiheadAttention` to train
    mode and disables dropout to avoid the unsupported NPU fused operator.
    
    Args:
        model: PyTorch model.
        enable: Whether to enable this workaround, usually controlled by args.
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
