# syntax = docker/dockerfile:1.3
# this ^^ line is very important as it enables support for --mount=type=cache
# ref: https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/syntax.md#run---mounttypecache

## --- globals 

# the following arguments aredefined via ARG to use later in other base images
# ref: https://github.com/moby/moby/issues/37345
ARG POETRY_HOME="/opt/poetry"
ARG PYSETUP_PATH="/opt/pysetup"


## --- poetry base image
FROM python:3.9-slim as base

ARG POETRY_HOME
ARG PYSETUP_PATH
# python env
ENV \
    ## - python
    PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    ## - pip
    PIP_NO_CACHE_DIR=1 \
    \
    ## - poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    # make poetry install to this location
    POETRY_HOME=$POETRY_HOME \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # poetry specific cache
    POETRY_CACHE_DIR="/opt/poetry/.cache" \
    \
    ## - paths
    # this is where our requirements + virtual environment will live
    PYSETUP_PATH=$PYSETUP_PATH
# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$PYSETUP_PATH/.venv/bin:$PATH"

# install curl and essentials
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    # deps for installing poetry
    curl
# deps for building python deps
# build-essential
# install poetry
# by default it respects $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python -


## --- build stage
FROM base AS build

WORKDIR $PYSETUP_PATH

COPY poetry.lock pyproject.toml ./

# install runtime deps
# uses $POETRY_VIRTUALENVS_IN_PROJECT internally
# it also uses poetry caching, thanks @cbrochtrup
# ref: https://github.com/python-poetry/poetry/issues/3374#issuecomment-857878275
RUN --mount=type=cache,target=$POETRY_CACHE_DIR/cache \
    --mount=type=cache,target=$POETRY_CACHE_DIR/artifacts \
    poetry install --no-dev --no-root


## --- production image
FROM python:3.9-slim as production

ARG PYSETUP_PATH
ENV PATH="$PYSETUP_PATH/.venv/bin:$PATH"
WORKDIR /app

COPY --from=build $PYSETUP_PATH $PYSETUP_PATH
COPY . .
COPY pyproject.toml README.md ./

# we install the current package with pip so that
# we can access their commands
RUN pip install --no-deps .

ENTRYPOINT python sumare/launch.py

EXPOSE 7680