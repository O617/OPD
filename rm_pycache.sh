find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 从git缓存中移除已跟踪的pyc文件
git rm -r --cached .
git add .