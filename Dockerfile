# 1. 选择基础镜像 (根据刚才 python --version 的结果)
# "slim" 版本体积更小，推荐生产使用
FROM python:3.12-slim

# 2. 设置工作目录
WORKDIR /app/rl-benchmarking

# 3. 设置环境变量 (防止 Python 生成 .pyc 文件，并让日志直接输出)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. 复制依赖文件到容器
COPY requirements.txt .

# 5. 安装依赖
# --no-cache-dir 可以减小镜像体积
# -i https://pypi.tuna.tsinghua.edu.cn/simple 是使用清华源加速（国内推荐）
RUN pip install uv==0.9.21
RUN uv pip install --no-cache-dir -r requirements.txt

# 6. 复制当前目录下的所有代码到容器
COPY . .

# 7. 指定启动命令 (修改为你的主程序入口)
CMD ["python", "main.py"]
