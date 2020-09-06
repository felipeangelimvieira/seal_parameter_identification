FROM python:3.7-slim
RUN pip install jupyter jupyterlab numpy scipy jax matplotlib jaxlib tqdm pandas && \
    apt-get update && apt-get install -y nodejs npm
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager



