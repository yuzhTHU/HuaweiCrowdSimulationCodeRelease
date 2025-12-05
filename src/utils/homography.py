import numpy as np
from scipy.interpolate import griddata

def calc_homography_mat(src, dst):
    """
    计算透视变换矩阵 H，使得 dst ~ H * src。
    
    使用直接线性变换 (DLT) 算法和 SVD 分解求解。通常用于将图像像素坐标
    映射到真实世界的物理坐标。

    Args:
        src (np.ndarray): 源平面上的四个点坐标。
            Shape: (4, 2)
        dst (np.ndarray): 目标平面上对应的四个点坐标。
            Shape: (4, 2)

    Returns:
        np.ndarray: 计算得到的 3x3 单应性矩阵 (Homography Matrix)。
            Shape: (3, 3)
    
    Raises:
        ValueError: 如果输入点的形状不符合 (4, 2)。
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
    使用透视变换矩阵 H 对点集 src 进行坐标变换。
    
    将二维点集转换为齐次坐标，应用矩阵乘法，再转换回二维直角坐标。

    Args:
        src (np.ndarray): 待变换的源点集。
            Shape: (..., 2) 最后一维为 (x, y)。
        H (np.ndarray): 透视变换矩阵。
            Shape: (3, 3)

    Returns:
        np.ndarray: 变换后的目标点集。
            Shape: (..., 2) 与输入形状一致。
    """
    ones = np.ones((*src.shape[:-1], 1), dtype=np.float32) # (..., 1)
    xy1 = np.concatenate([src, ones], axis=-1)  # (..., 3)
    tgt = xy1 @ H.T  # (..., 3)
    tgt = tgt[..., :2] / tgt[..., (2,)]  # 除以最后一维以归一化 (..., 2)
    return tgt

def image_to_world(src, H, dot_per_meter=5):
    """
    将图像数据变换到世界坐标系，并重新栅格化为规则网格。
    
    该函数首先将图像像素坐标映射到世界坐标，然后使用双线性插值
    (griddata) 将不规则的散点数据重采样到物理尺寸均匀的网格上。
    
    Args:
        src (np.ndarray): 输入图像数据（如地图语义掩码）。
            Shape: (W, H, ...) 第一维对应 x 轴，第二维对应 y 轴。
        H (np.ndarray): 从图像坐标到世界坐标的变换矩阵。
            Shape: (3, 3)
        dot_per_meter (int, optional): 输出网格的分辨率（每米采样点数）。默认为 5。

    Returns:
        tuple: 包含以下元素的元组:
            - map (np.ndarray): 重采样后的栅格化地图。
            - xmin (float): 地图在世界坐标系下的 X 轴下界。
            - xmax (float): 地图在世界坐标系下的 X 轴上界。
            - ymin (float): 地图在世界坐标系下的 Y 轴下界。
            - ymax (float): 地图在世界坐标系下的 Y 轴上界。
    """
    w, h = src.shape[:2]
    value = src.reshape(w*h, *src.shape[2:])  # (w*h, ...)

    ii, jj = np.meshgrid(np.arange(w), np.arange(h), indexing="ij") # (w, h)
    src_coord = np.stack([ii, jj], axis=-1).astype(np.float32)  # (w, h, 2)
    tgt_coord = affine_transformation(src_coord, H)  # (w, h, 2)
    x = tgt_coord[..., 0].reshape(-1)
    y = tgt_coord[..., 1].reshape(-1)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax+1/dot_per_meter, 1/dot_per_meter)[:-1],
        np.arange(ymin, ymax+1/dot_per_meter, 1/dot_per_meter)[:-1],
        indexing='ij',
    )
    map = griddata(
        points=np.stack([x, y], axis=-1),
        values=value,
        xi=(xx, yy),
        method="linear",  # 'linear' 对应双线性插值
    )
    return map, xmin, xmax, ymin, ymax
