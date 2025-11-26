FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    clang \
    clang-tidy \
    clang-format \
    cppcheck \
    valgrind \
    python3 \
    python3-pip \
    ca-certificates \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Create a non-root user to run evaluations
RUN useradd -m -s /bin/bash runner
WORKDIR /home/runner
USER runner

# Entrypoint will be a script that runs evaluation commands mounted from host
COPY docker/entrypoint.sh /home/runner/entrypoint.sh
RUN chmod +x /home/runner/entrypoint.sh
ENTRYPOINT ["/home/runner/entrypoint.sh"]
