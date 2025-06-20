# vllm build

```bash
# 初始化 vllm 相关开发环境
python -m venv vllmenv
source vllmenv/bin/activate
pip install vllm

# build
python setup.py bdist_wheel
```

# bladellm build

在 bladellm 开发镜像中

```bash
python setup.py bdist_wheel
```
