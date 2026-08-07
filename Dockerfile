# syntax=docker/dockerfile:1

# --------------------------------------------------------------- web build --
FROM node:22-slim AS web

# Substituted into index.html's Open Graph tags. Left empty the image URLs stay
# root-relative, which is legal and worse: a scraper is not obliged to resolve
# them, so a shared link loses its preview.
ARG VITE_SITE_URL=""
ENV VITE_SITE_URL=$VITE_SITE_URL

WORKDIR /build

# Copied before the sources so editing a component does not reinstall the tree.
COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
# `npm run build` is `tsc --noEmit && vite build`, so a type error fails the
# image rather than shipping.
RUN npm run build

# ----------------------------------------------------------------- runtime --
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# numpy, scipy and matplotlib all publish manylinux wheels, so no compiler is
# needed and none is installed.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install .

COPY --from=web /build/dist ./web/dist

# One shared core runs two job workers. OpenBLAS defaulting to a thread per
# core underneath them only makes them contend.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    ATOMSIM_WEB_DIST=/app/web/dist \
    ATOMSIM_CLIENT_IP_HEADER=fly-client-ip

RUN useradd --create-home --uid 1000 atomsim \
    && mkdir -p /tmp/matplotlib \
    && chown atomsim:atomsim /tmp/matplotlib
USER atomsim

EXPOSE 8080

# Not `atomsim serve`: that binds 127.0.0.1, which is unreachable from outside
# the container, and opens a browser that does not exist here.
CMD ["uvicorn", "atomsim.server.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8080"]
