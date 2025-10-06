import numpy as np
from scipy.interpolate import griddata

def calc_homography_mat(src, dst):
    """
    计算透视变换矩阵 H 使得 dst ~ H * src
    src: shape (4, 2)
    dst: shape (4, 2)
    返回 H: shape (3, 3)
    """
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("src and dst must be (4, 2) arrays")

    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([-x, -y, -1,  0,  0,  0, x*X, y*X, X])
        A.append([ 0,  0,  0, -x, -y, -1, x*Y, y*Y, Y])
    A = np.array(A, dtype=np.float64)

    # 求解最小特征值对应的特征向量 (SVD)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :] / Vt[-1, -1]  # 归一化最后一个参数为 1
    H = h.reshape(3, 3)
    return H

def affine_transformation(src, H):
    """
    使用透视变换矩阵 H 变换点集 src
    src: shape (..., 2)
    H: shape (3, 3)
    返回 dst: shape (..., 2)
    """
    ones = np.ones((*src.shape[:-1], 1), dtype=np.float32) # (..., 1)
    xy1 = np.concatenate([src, ones], axis=-1)  # (..., 3)
    tgt = xy1 @ H.T  # (..., 3)
    tgt = tgt[..., :2] / tgt[..., (2,)]  # 除以最后一维以归一化 (..., 2)
    return tgt

def image_to_world(src, H, dot_per_meter=5):
    """
    使用透视变换矩阵 H 变换点集 src 并双线性插值为 2D 网格
    src: shape (h, w, ...)
    H: shape (3, 3)
    返回 dst: shape (h, w, ...)
    """
    h, w = src.shape[:2]
    value = src.reshape(h*w, *src.shape[2:])  # (h*w, ...)

    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij") # (h, w)
    src_coord = np.stack([ii, jj], axis=-1).astype(np.float32)  # (h, w, 2)
    tgt_coord = affine_transformation(src_coord, H)  # (h, w, 2)
    x = tgt_coord[..., 0].reshape(-1)
    y = tgt_coord[..., 1].reshape(-1)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax+1/dot_per_meter, 1/dot_per_meter)[:-1],
        np.arange(ymin, ymax+1/dot_per_meter, 1/dot_per_meter)[:-1],
    )
    map = griddata(
        points=np.stack([x, y], axis=-1),
        values=value,
        xi=(xx, yy),
        method="linear",  # 'linear' 对应双线性插值
    )
    return map, xmin, xmax, ymin, ymax
