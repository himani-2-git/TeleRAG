# TeleRAG bot image. CPU only: no GPU libraries are installed.
#
#   docker build -t telerag .
#
# The bot runs as a normal user (not root). Data lives in /app/data and logs in /app/logs,
# which docker-compose.yml mounts as volumes so they survive container restarts.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

# A user without a login shell or admin rights, and the folders it may write to.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/logs /opt/hf \
    && chown -R app:app /app /opt/hf

WORKDIR /app

# PyTorch first, from the CPU-only index. The default PyPI build of torch for Linux drags in
# several GB of CUDA libraries that this project never uses. The version is the one the tests
# were run with.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1

# The other libraries, pinned to the versions the tests were run with.
COPY requirements/base.txt requirements/docker-constraints.txt requirements/
RUN pip install -r requirements/base.txt -c requirements/docker-constraints.txt

# Download the embedding model now, so the container works without internet after the build
# and the first question is not slow. Keep this value equal to EMBEDDING_MODEL in .env.
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN python -c "import sys; from sentence_transformers import SentenceTransformer; SentenceTransformer(sys.argv[1])" "$EMBEDDING_MODEL" \
    && chown -R app:app /opt/hf

COPY --chown=app:app main.py ./
COPY --chown=app:app src/ src/
COPY --chown=app:app prompts/ prompts/

USER app

# python-telegram-bot stops cleanly on SIGTERM (docker stop): it finishes running questions.
CMD ["python", "main.py"]
