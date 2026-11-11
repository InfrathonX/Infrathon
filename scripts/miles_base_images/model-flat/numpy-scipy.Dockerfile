FROM radixark/miles@sha256:45ae7833f97a01369dbd8f336784830a87d94587bc010c7b8065291cc15cd7ca
RUN python -m pip install --no-deps numpy==1.26.4 scipy==1.15.3
LABEL mlsys-authoring="miles-model-validation"
