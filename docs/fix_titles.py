import os

# 配置：文件名 -> 中文标题
TITLE_MAPPING = {
    "src.dataset.rst": "数据加载与处理 (Dataset)",
    "src.model.rst": "核心预测模型 (Model)",
    "src.diffusion.rst": "扩散生成算法 (Diffusion)",
    "src.utils.rst": "通用工具组件 (Utils)",
    "src.rst": "源码根目录 (Source Root)"
}

SOURCE_DIR = "./source"  # 根据您的实际路径调整，如果在 docs 下运行则是 ./source

def replace_first_line(filepath, new_title):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 修改第一行标题
    lines[0] = new_title + "\n"
    
    # 修改第二行下划线，使其长度与标题匹配（ReST 格式要求）
    # 注意：中文字符在视觉上是宽字符，但在 len() 中算 1，
    # 为了保险，通常生成一个足够长的下划线，或者粗略计算宽度
    # 这里简单地用 len(new_title.encode('gbk')) 来估算显示宽度，或者直接给个固定的长线
    underline_len = max(len(new_title) * 2, 20) 
    lines[1] = "=" * underline_len + "\n"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"✅ 已修改标题: {filepath} -> {new_title}")

def main():
    for filename, title in TITLE_MAPPING.items():
        filepath = os.path.join(SOURCE_DIR, filename)
        if os.path.exists(filepath):
            replace_first_line(filepath, title)
        else:
            print(f"⚠️ 文件未找到: {filepath}")

if __name__ == "__main__":
    main()