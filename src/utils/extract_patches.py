import torch
import numpy as np
import torch.nn.functional as F
from numpy.lib.stride_tricks import as_strided


def extract_patches_numpy(arr, idx, jdx, r=10):
    """
    Efficiently extract local square patches centered at `(idx, jdx)` from a
    2D array.

    Out-of-bound regions are automatically filled with NaN.

    Args:
        arr (torch.Tensor): Input 2D map tensor.
            Shape: (H, W)
        idx (torch.LongTensor): Row indices of the centers, corresponding to
            the map x-coordinate. Shape: `(..., N)`.
        jdx (torch.LongTensor): Column indices of the centers, corresponding to
            the map y-coordinate. Shape must match `idx`.
        r (int, optional): Extraction radius. The window size is `(2r+1)x(2r+1)`.

    Returns:
        torch.Tensor: Stacked extracted patches.
            Shape: (..., N, 2r+1, 2r+1)
    """
    # 1. Type check: the array must be floating-point so NaN padding can be used.
    # This can be skipped for maximum speed if the input is guaranteed to be float already.
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(float)
    # 2. Padding.
    # Pad `r` NaNs on all four sides.
    # The original `(0, 0)` now lives at `(r, r)` in `padded`.
    padded = np.pad(arr, pad_width=r, mode='constant', constant_values=np.nan)
    # 3. Build a strided view.
    # We do not want `view[i, j]` to be the window whose top-left corner is `(i, j)`;
    # we want the window centered at the original `(i, j)`.
    # Original `(i, j)` corresponds to `(i+r, j+r)` in `padded`.
    # The window range is `[i+r-r : i+r+r+1] -> [i : i+2r+1]`.
    # Therefore, if we build a view sliding from `padded[0, 0]`,
    # then `view[i, j]` is exactly `padded[i:i+2r+1, j:j+2r+1]`.
    # This creates the view instantly without copying data.
    # Note that the shape must be limited by `arr.shape`, otherwise view indexing goes out of bounds.
    # In practice, we only index with `idx` and `jdx`, so they just need to remain valid.
    # Output view shape: `(original_rows, original_cols, window_height, window_width)`.
    s_row, s_col = padded.strides
    window_size = 2 * r + 1
    subs = as_strided(
        padded,
        shape=(arr.shape[0], arr.shape[1], window_size, window_size),
        strides=(s_row, s_col, s_row, s_col),
    )
    # 4. Use fancy indexing to extract the result.
    # This step copies memory and produces the final result, e.g. `(l, 21, 21)`.
    return subs[idx, jdx]


def extract_patches_torch(arr, idx, jdx, r=10):
    """
    Args:
        arr: `(H, W)` input 2D tensor.
        idx, jdx: `(..., l)` center-coordinate LongTensors with any leading dimensions.
        r: Radius, default `10`, so the window size is `21x21`.

    Returns:
        A tensor of shape `(..., l, 2r+1, 2r+1)` with NaN-filled out-of-bound regions.
    """
    # 1. Ensure `arr` is floating-point so NaN padding can be used.
    if not arr.is_floating_point():
        arr = arr.float()
        
    # 2. Preprocess padding.
    # The `F.pad` argument order is `(Left, Right, Top, Bottom)`.
    # The original `(0, 0)` becomes `(r, r)` in `padded`.
    padded = F.pad(arr, (r, r, r, r), mode='constant', value=float('nan'))
    
    # 3. Build relative offsets for the local window.
    # `rows_offset`: `(2r+1, 1)` -> broadcast across columns.
    # `cols_offset`: `(1, 2r+1)` -> broadcast across rows.
    rows_offset = r + torch.arange(-r, r + 1, device=arr.device).view(-1, 1)
    cols_offset = r + torch.arange(-r, r + 1, device=arr.device).view(1, -1)
    
    # 4. Compute the absolute coordinate grid with broadcasting.
    # Assume `idx` has shape `(Batch, l)` -> expand to `(Batch, l, 1, 1)`.
    # The final grid shape is `(Batch, l, 2r+1, 2r+1)`.
    # First expand `idx/jdx` to the last two dimensions.
    # Then compute positions in the padded coordinate system: original coordinate + r + offset.
    grid_rows = idx.unsqueeze(-1).unsqueeze(-1) + rows_offset
    grid_cols = jdx.unsqueeze(-1).unsqueeze(-1) + cols_offset
    
    # 5. Safety handling: clamp.
    # This avoids CUDA errors from extreme out-of-range indices.
    # As long as clamping keeps indices inside `padded`, the result is either valid data
    # or NaN from the padding.
    # For example, if `idx=-1000`, then after adding `r` it is still negative, so clamping to 0
    # yields `padded[0, :]`, which is NaN as expected.
    grid_rows = torch.clamp(grid_rows, 0, padded.shape[0] - 1)
    grid_cols = torch.clamp(grid_cols, 0, padded.shape[1] - 1)
    
    # 6. Gather / indexing.
    # Extract directly with advanced indexing.
    patches = padded[grid_rows, grid_cols]
    return patches
