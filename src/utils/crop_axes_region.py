import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def crop_axes_region(ax, xmin, xmax, ymin, ymax, pad_pixels=0, dpi=500):
    """
    从一个 matplotlib Axes 中裁剪指定数据区域，返回 PIL.Image。
    
    参数:
        ax: matplotlib.axes.Axes 对象
        xmin, xmax, ymin, ymax: 数据坐标范围
        pad_pixels: 裁剪时向内缩进像素，避免边线干扰
    
    返回:
        PIL.Image 对象
    """
    fig = ax.figure
    canvas = fig.canvas
    raw_dpi = ax.figure.dpi
    fig.set_dpi(dpi)   # 改变 Figure 的 DPI

    canvas.draw()  # 渲染 Figure
    argb = canvas.tostring_argb()
    w, h = canvas.get_width_height()

    # 坐标 -> 像素转换
    def data_to_pixel(xdata, ydata):
        px, py = ax.transData.transform(np.array([[xdata, ydata]]))[0]
        return int(round(px)), int(round(h - py))  # y 翻转
    
    pxmin, pymin = data_to_pixel(xmin, ymin)
    pxmax, pymax = data_to_pixel(xmax, ymax)

    fig.set_dpi(raw_dpi)  # 恢复原始 DPI

    # Figure 渲染成 RGBA 数组
    buf = np.frombuffer(argb, dtype=np.uint8)
    buf = buf.reshape(h, w, 4)
    buf = buf[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA
    
    # 裁剪区域，加入 pad
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
    # 示例用法
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots(dpi=100)
    ax.axis('equal')
    ax.plot(x, y, lw=1)

    # 红框仅作参考
    xmin, xmax, ymin, ymax = 2, 5, -0.5, 0.5
    ax.plot([xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            color='red', linewidth=0.5, antialiased=False)

    crop_axes_region(ax, xmin, xmax, ymin, ymax, pad_pixels=0, dpi=800)