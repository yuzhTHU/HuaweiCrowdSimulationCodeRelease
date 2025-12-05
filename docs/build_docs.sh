#!/bin/bash

# 获取脚本文件所在的绝对目录 (即 docs/ 目录)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 切换工作目录到 docs/
cd "$SCRIPT_DIR"

# 1. 清理旧构建
# 检查 Makefile 是否存在，防止误操作
if [ -f "Makefile" ]; then
    make clean
else
    echo "Error: Makefile not found. Are you sure the script is in the 'docs/' folder?"
    exit 1
fi

# 2. 自动生成 API rst 文件 (使用 -f 覆盖)
# 注意：此时应当已经位于 docs/ 目录下
echo "Generating API .rst files via sphinx-apidoc..."
sphinx-apidoc -f -e -M -o source/ ../src

# 3. 运行脚本修正标题
# 检查 fix_titles.py 是否存在
if [ -f "fix_titles.py" ]; then
    echo "Running fix_titles.py to beautify headers..."
    python fix_titles.py
else
    echo "Warning: fix_titles.py not found in $(pwd), skipping title fix."
fi

# 4. 生成 HTML
echo "Building HTML..."
make html

# 5. 提示完成 (输出绝对路径方便点击)
BUILD_DIR="$SCRIPT_DIR/build/html/index.html"


echo "========================================"
echo "文档生成完毕！"
echo "请在浏览器打开: file://$BUILD_DIR"
echo "您也可以使用以下命令启动本地服务器以查看文档："
echo "  python -m http.server 8000 -d $SCRIPT_DIR/build/html"
echo "然后在浏览器中访问 http://localhost:8000"
echo "========================================"
