FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# install basics
RUN set -ex ; \
    apt-get update ; \
    apt-get install -y \
        autoconf \
        cmake \
        curl \
        g++ \
        pkg-config \
        git \
	mpich \
	libblas-dev \
	liblapack-dev \
        libmpich-dev \
        openmpi-bin \
        libopenmpi-dev \
        python3 \
        python3-dev \
        python3-pip \
        unzip \
        rsync \
	sudo \
        openssh-client \
        openssh-server \
        valgrind \
        vim \
        wget \
        xterm

# install OneAPI HPC kit
# repository to install Intel(R) oneAPI Libraries
#RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | apt-key add -
#RUN echo "deb [trusted=yes] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list
#RUN apt-get update && \
#  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#    curl ca-certificates gpg-agent software-properties-common && \
#  rm -rf /var/lib/apt/lists/*
#RUN apt-get update && \
#  apt-get install -y --no-install-recommends \
#  ca-certificates build-essential pkg-config gnupg libarchive13 \
#  intel-basekit-getting-started intel-oneapi-common-licensing intel-oneapi-common-vars \
#  intel-oneapi-mkl-devel intel-hpckit-getting-started && \
#  rm -rf /var/lib/apt/lists/*

# install libxsmm
#ARG libxsmm_version=1.17

#RUN set -ex ; \
#    curl -LsS https://github.com/hfp/libxsmm/archive/${libxsmm_version}.tar.gz | tar -xz -C /opt ; \
#    ln -s libxsmm-${libxsmm_version} /opt/libxsmm ; \
#    make -j -C /opt/libxsmm MALLOC=0

#ENV PKG_CONFIG_PATH="/opt/libxsmm/lib:${PKG_CONFIG_PATH}"

ENV OMPI_ALLOW_RUN_AS_ROOT=1 
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 
ENV HYDRA_HOST_FILE=/etc/hydra-hostfile

RUN useradd -ms /bin/bash dev 
RUN echo "dev:pass" | chpasswd

RUN mkdir /run/sshd
#RUN sevice ssh stop

RUN sed -i 's/.*\(StrictHostKeyChecking\).*/\1 no/' /etc/ssh/ssh_config && \
    sed -i 's/session\s*required\s*pam_loginuid.so/session optional pam_loginuid.so/' /etc/pam.d/sshd && \
    yes 'yes' | ssh-keygen -t rsa -P "" -f /etc/ssh/ssh_host_rsa_key && \
    mkdir /home/dev/.ssh && \
    ssh-keygen -t rsa -P "" -f /home/dev/.ssh/id_rsa && \
    cat /home/dev/.ssh/id_rsa.pub > /home/dev/.ssh/authorized_keys && \
    chmod 750 /home/dev/.ssh && chmod 640 /home/dev/.ssh/authorized_keys && \
    chmod 600 /home/dev/.ssh/id_rsa /home/dev/.ssh/id_rsa.pub && \
    chown -R dev: /home/dev/.ssh

WORKDIR /home/dev

RUN pip3 install numpy
