import numpy as np
from scipy.interpolate import griddata

def calc_homography_mat(src, dst):
    """
    Compute the homography matrix `H` such that `dst ~ H * src`.

    Solved with the Direct Linear Transform (DLT) algorithm and SVD. Commonly
    used to map image pixel coordinates into real-world coordinates.

    Args:
        src (np.ndarray): Four source-plane points.
            Shape: (4, 2)
        dst (np.ndarray): Four corresponding target-plane points.
            Shape: (4, 2)

    Returns:
        np.ndarray: The computed `3x3` homography matrix.
            Shape: (3, 3)
    
    Raises:
        ValueError: If the input points are not shaped `(4, 2)`.
    """
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("src and dst must be (4, 2) arrays")

    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([-x, -y, -1,  0,  0,  0, x*X, y*X, X])
        A.append([ 0,  0,  0, -x, -y, -1, x*Y, y*Y, Y])
    A = np.array(A, dtype=np.float64)

    # Solve with SVD using the singular vector associated with the smallest singular value.
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :] / Vt[-1, -1]  # Normalize the last coefficient to 1.
    H = h.reshape(3, 3)
    return H

def affine_transformation(src, H):
    """
    Transform point set `src` with homography matrix `H`.

    Converts 2D points to homogeneous coordinates, applies the matrix
    multiplication, then converts back to Cartesian coordinates.

    Args:
        src (np.ndarray): Source points to transform.
            Shape: `(..., 2)` where the last dimension is `(x, y)`.
        H (np.ndarray): Homography matrix.
            Shape: (3, 3)

    Returns:
        np.ndarray: Transformed target points with the same shape as input.
    """
    ones = np.ones((*src.shape[:-1], 1), dtype=np.float32) # (..., 1)
    xy1 = np.concatenate([src, ones], axis=-1)  # (..., 3)
    tgt = xy1 @ H.T  # (..., 3)
    tgt = tgt[..., :2] / tgt[..., (2,)]  # Divide by the last coordinate for normalization.
    return tgt

def image_to_world(src, H, dot_per_meter=5):
    """
    Transform image data into world coordinates and resample it onto a regular grid.

    The function first maps image pixel coordinates into world coordinates, then
    uses bilinear interpolation via `griddata` to resample irregular points
    onto a grid with uniform physical spacing.
    
    Args:
        src (np.ndarray): Input image data, e.g. a semantic map mask.
            Shape: `(W, H, ...)`, where the first dimension is `x` and the
            second is `y`.
        H (np.ndarray): Transformation matrix from image coordinates to world coordinates.
            Shape: (3, 3)
        dot_per_meter (int, optional): Output grid resolution in samples per meter.

    Returns:
        tuple: A tuple containing:
            - map (np.ndarray): Resampled rasterized map.
            - xmin (float): Lower x bound in world coordinates.
            - xmax (float): Upper x bound in world coordinates.
            - ymin (float): Lower y bound in world coordinates.
            - ymax (float): Upper y bound in world coordinates.
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
        method="linear",  # `linear` corresponds to bilinear interpolation here.
    )
    return map, xmin, xmax, ymin, ymax
