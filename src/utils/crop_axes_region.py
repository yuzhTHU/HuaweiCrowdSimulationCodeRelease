import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def crop_axes_region(ax, xmin, xmax, ymin, ymax, pad_pixels=0, dpi=500):
    """
    Crop a specified data region from a matplotlib `Axes` and return it as a
    `PIL.Image`.

    Args:
        ax: `matplotlib.axes.Axes` object.
        xmin, xmax, ymin, ymax: Data-coordinate bounds.
        pad_pixels: Inward pixel padding to avoid border artifacts.

    Returns:
        `PIL.Image` object.
    """
    fig = ax.figure
    canvas = fig.canvas
    raw_dpi = ax.figure.dpi
    fig.set_dpi(dpi)   # Change the figure DPI.

    canvas.draw()  # Render the figure.
    argb = canvas.tostring_argb()
    w, h = canvas.get_width_height()

    # Coordinate -> pixel conversion.
    def data_to_pixel(xdata, ydata):
        px, py = ax.transData.transform(np.array([[xdata, ydata]]))[0]
        return int(round(px)), int(round(h - py))  # Flip y.
    
    pxmin, pymin = data_to_pixel(xmin, ymin)
    pxmax, pymax = data_to_pixel(xmax, ymax)

    fig.set_dpi(raw_dpi)  # Restore the original DPI.

    # Render the figure into an RGBA array.
    buf = np.frombuffer(argb, dtype=np.uint8)
    buf = buf.reshape(h, w, 4)
    buf = buf[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA
    
    # Crop the region and apply padding.
    x0, x1 = sorted([pxmin, pxmax])
    y0, y1 = sorted([pymin, pymax])
    
    x0 += pad_pixels
    x1 -= pad_pixels
    y0 += pad_pixels
    y1 -= pad_pixels
    
    cropped = buf[y0:y1, x0:x1]
    return Image.fromarray(cropped)

if __name__ == "__main__":
    # ----------------------------
    # Example usage.
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots(dpi=100)
    ax.axis('equal')
    ax.plot(x, y, lw=1)

    # Red box for reference only.
    xmin, xmax, ymin, ymax = 2, 5, -0.5, 0.5
    ax.plot([xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            color='red', linewidth=0.5, antialiased=False)

    crop_axes_region(ax, xmin, xmax, ymin, ymax, pad_pixels=0, dpi=800)
