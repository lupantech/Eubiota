set -ex

# python -m uv pip install --upgrade uv pip

uv pip install --no-cache-dir packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
uv pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-cache-dir transformers==4.53.3
uv pip install --no-cache-dir flash-attn==2.8.1 --no-build-isolation
uv pip install --no-cache-dir vllm==0.9.2
uv pip install --no-cache-dir verl==0.5.0

uv pip install --no-cache-dir -e .[dev,agent]
