FROM python:3.12.11-slim-bookworm@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49

RUN pip install --no-cache-dir aiohttp==3.12.15
COPY model-api/entrypoint.sh model-api/server.py model-api/transport.py \
     backend-serving-contract.json /opt/model-api/
RUN chmod 0555 \
      /opt/model-api/entrypoint.sh \
      /opt/model-api/server.py \
      /opt/model-api/transport.py \
      /opt/model-api/backend-serving-contract.json

EXPOSE 8100
ENTRYPOINT ["/opt/model-api/entrypoint.sh"]
