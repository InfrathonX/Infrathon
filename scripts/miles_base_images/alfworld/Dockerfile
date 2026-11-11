# Miles ALFWorld task base: miles-model-dependencies-flat:local plus alfworld 0.4.2 and the
# packages it needs that the base lacks (nothing already in the base is replaced).
# Replaces the original recipe that COPYed a pre-built site-packages directory: the same packages
# are installed here from requirements.txt in a builder stage.
ARG BASE_IMAGE=miles-model-dependencies-flat:local
FROM ${BASE_IMAGE} AS build
USER root
COPY requirements.txt /tmp/alfworld-requirements.txt
RUN python -m venv --system-site-packages /tmp/alfw \
    && /tmp/alfw/bin/pip install --no-cache-dir --no-deps -r /tmp/alfworld-requirements.txt \
    && mkdir /tmp/site \
    && cd /tmp/alfw/lib/python3.12/site-packages \
    && find . -mindepth 1 -maxdepth 1 ! -name 'pip' ! -name 'pip-*.dist-info' ! -name '__pycache__' \
         -exec cp -a '{}' /tmp/site/ ';'

FROM ${BASE_IMAGE}
USER root
COPY --from=build /tmp/site/ /usr/local/lib/python3.12/dist-packages/
RUN python -c "import alfworld, textworld, numpy, torch, sglang; from alfworld.agents.environment.alfred_tw_env import AlfredDemangler; print('alfworld ok', numpy.__version__, torch.__version__)"
