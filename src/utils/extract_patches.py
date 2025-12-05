import torch
import numpy as np
import torch.nn.functional as F
from numpy.lib.stride_tricks import as_strided


def extract_patches_numpy(arr, idx, jdx, r=10):
    """
    从二维张量中高效提取以 (idx, jdx) 为中心的局部方形图块 (Patches)。
    
    使用 `torch.nn.functional.pad` 和高级索引实现，支持 GPU 加速和自动梯度。
    越界区域会自动填充 NaN。

    Args:
        arr (torch.Tensor): 输入的二维大地图张量。
            Shape: (H, W)
        idx (torch.LongTensor): 中心点的行索引 (对应 map 的 x 坐标)。
            Shape: (..., N) 任意维度。
        jdx (torch.LongTensor): 中心点的列索引 (对应 map 的 y 坐标)。
            Shape: (..., N) 维度需与 idx 一致。
        r (int, optional): 提取半径。窗口大小将为 (2r+1)x(2r+1)。默认为 10。

    Returns:
        torch.Tensor: 提取出的局部图块堆叠。
            Shape: (..., N, 2r+1, 2r+1)
    """
    # 1. 数据类型检查：为了填充 NaN，数组必须是浮点型
    # 如果为了极致速度且确定原数据已是 float，可跳过此步
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(float)
    # 2. Padding (填充)
    # 在上下左右各填充 r 个 NaN
    # 原始 (0,0) 现在位于 padded 的 (r, r)
    padded = np.pad(arr, pad_width=r, mode='constant', constant_values=np.nan)
    # 3. 构建 Strided View (虚拟视图)
    # 我们希望 view[i, j] 对应的是以原始 (i,j) 为左上角的窗口吗？
    # 不，我们希望对应以原始 (i,j) 为 *中心* 的窗口。
    # 原始 (i,j) 对应 padded 中的 (i+r, j+r)。
    # 窗口范围是 [i+r-r : i+r+r+1] -> [i : i+2r+1]。
    # 所以，如果我们构建一个从 padded[0,0] 开始滑动的视图，
    # 那么 view[i, j] 恰好就是 padded[i:i+2r+1, j:j+2r+1]。
    # 创建视图 (瞬间完成，不复制数据)
    # 注意：这里的 shape 必须限制在 arr.shape 大小，否则 view 索引会越界
    # 但实际上我们只索引 idx, jdx，只要它们在合法范围内即可
    # 输出视图的形状：(原行数, 原列数, 窗口高, 窗口宽)
    s_row, s_col = padded.strides
    window_size = 2 * r + 1
    subs = as_strided(
        padded,
        shape=(arr.shape[0], arr.shape[1], window_size, window_size),
        strides=(s_row, s_col, s_row, s_col),
    )
    # 4. 利用花式索引提取结果
    # 这一步会发生内存复制，生成最终结果 (l, 21, 21)
    return subs[idx, jdx]


def extract_patches_torch(arr, idx, jdx, r=10):
    """
    参数:
    arr: (H, W) 输入的 2D Tensor
    idx, jdx: (..., l) 任意维度的中心坐标 LongTensor
    r: 半径 (默认10, 窗口大小为 21x21)
    
    返回:
    (..., l, 2r+1, 2r+1) 的 Tensor, 越界部分填充 NaN
    """
    # 1. 确保 arr 是浮点型 (为了填充 NaN)
    if not arr.is_floating_point():
        arr = arr.float()
        
    # 2. 预处理 Padding
    # F.pad 参数顺序: (Left, Right, Top, Bottom)
    # 原始 (0,0) 变成了 padded 中的 (r, r)
    padded = F.pad(arr, (r, r, r, r), mode='constant', value=float('nan'))
    
    # 3. 构建局部窗口的相对偏移量
    # rows_offset: (2r+1, 1) -> 广播到列
    # cols_offset: (1, 2r+1) -> 广播到行
    rows_offset = r + torch.arange(-r, r + 1, device=arr.device).view(-1, 1)
    cols_offset = r + torch.arange(-r, r + 1, device=arr.device).view(1, -1)
    
    # 4. 计算绝对坐标网格 (Broadcasting Magic)
    # idx 维度假设为 (Batch, l) -> 扩展为 (Batch, l, 1, 1)
    # 最终 grid 维度为 (Batch, l, 2r+1, 2r+1)
    
    # 先把 idx/jdx 扩展到最后两维
    # 计算在 padded 坐标系中的位置 (原坐标 + r + 偏移)
    grid_rows = idx.unsqueeze(-1).unsqueeze(-1) + rows_offset
    grid_cols = jdx.unsqueeze(-1).unsqueeze(-1) + cols_offset
    
    # 5. 安全性处理：Clamp (钳位)
    # 防止极度越界导致 CUDA error。
    # 只要 clamp 在 padded 的范围内，取出的就是 padding 填充的 NaN，或者是有效数据。
    # 比如 idx=-1000, +r 后还是负数，clamp 到 0，取出的 padded[0,:] 是 NaN。符合预期。
    grid_rows = torch.clamp(grid_rows, 0, padded.shape[0] - 1)
    grid_cols = torch.clamp(grid_cols, 0, padded.shape[1] - 1)
    
    # 6. Gather / Indexing
    # 直接使用高级索引提取
    patches = padded[grid_rows, grid_cols]
    return patches
