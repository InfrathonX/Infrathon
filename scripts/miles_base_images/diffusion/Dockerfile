# Pinned dependency base shared by the miles_diffusion benchmark tasks (md-opd-v1 / md-wandance-v1).
# Derived from miles_diffusion@29fde6ee docker/Dockerfile with every mutable input pinned:
#   base image by digest, sglang-miles-h3 by commit, FA3 wheel by sha256, hpsv2 by commit.
# miles_diffusion itself is NOT installed here: each task copies its own `git archive` export.
ARG SGLANG_BASE
FROM ${SGLANG_BASE}
ARG SGLANG_COMMIT=859a278eb882dddbc4ccab1d0a3697da04a66f28
ARG FA3_URL=https://github.com/yueming-yuan/miles-wheels/releases/download/cu129-x86_64/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl
ARG FA3_SHA256=b0f4d97418aa129522cd4b4e65ce516ddf8af64815f4ce040cb38a6d94cef971
WORKDIR /root/

RUN apt-get update \
    && apt-get install -y --no-install-recommends nvtop rsync dnsutils ethtool libglib2.0-0 libgl1 ripgrep jq procps \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/fa3 && curl -fSL -o /tmp/fa3/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl "${FA3_URL}" \
    && echo "${FA3_SHA256}  /tmp/fa3/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl" | sha256sum -c - \
    && pip install --no-deps /tmp/fa3/flash_attn_3-*.whl && python -c "import flash_attn_interface" && rm -rf /tmp/fa3

RUN cd /sgl-workspace/sglang \
    && git fetch --depth=1 https://github.com/sgl-project/sglang.git "${SGLANG_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${SGLANG_COMMIT}" \
    && SGLANG_BUILD_RUST_EXTS=none pip install -e python --no-deps
RUN pip install --no-deps "sglang-kernel==0.4.5+cu129" --extra-index-url https://docs.sglang.io/whl/cu129
RUN cd /sgl-workspace/sglang && kernels lock python && mkdir -p /root/.cache/sglang && mv python/kernels.lock /root/.cache/sglang/
RUN pip install torch_memory_saver==0.0.9

COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed PyJWT
RUN pip install -r /tmp/requirements.txt
RUN pip install --no-deps args==0.1.0 clint==0.5.1 "hpsv2 @ git+https://github.com/tgxs002/HPSv2@866735ecaae9" \
    && python3 -c "import hpsv2.src.open_clip"

# PaddleOCR en det/rec/cls weights (OCR reward runs on CPU); copied to the agent HOME below.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)"

# Provenance + hygiene: record resolved state, then drop VCS metadata and caches.
RUN mkdir -p /opt/miles-diffusion-base \
    && git -C /sgl-workspace/sglang rev-parse HEAD > /opt/miles-diffusion-base/sglang.commit \
    && pip freeze > /opt/miles-diffusion-base/pip-freeze.txt \
    && (pip check > /opt/miles-diffusion-base/pip-check.txt 2>&1 || true) \
    && python -c "import torch,sys; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list() if hasattr(torch.cuda,'get_arch_list') else '', sys.version)" > /opt/miles-diffusion-base/torch.txt \
    && sha256sum /root/.cache/sglang/kernels.lock /tmp/requirements.txt > /opt/miles-diffusion-base/inputs.sha256 \
    && find /sgl-workspace -type d -name .git -prune -exec rm -rf '{}' + \
    && rm -rf /root/.cache/pip /root/.cache/huggingface /root/.ssh /root/.config /tmp/requirements.txt \
    && (id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent) \
    && cp -a /root/.paddleocr /home/agent/.paddleocr && chown -R agent:agent /home/agent \
    && chmod 755 /root && chmod -R go-w /sgl-workspace/sglang \
    && git config --system safe.directory '*' \
    && python -c "import importlib.util as u; assert u.find_spec('miles') is None"
ENTRYPOINT []
CMD ["bash"]
