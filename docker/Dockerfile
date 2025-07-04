FROM ubuntu:22.04 AS builder

ENV BUILD_DIR="/appbuild"
RUN mkdir -p ${BUILD_DIR}

# Install essential packages
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update &&                                    \
    apt-get install --assume-yes --no-install-recommends \
      build-essential                                    \
      ca-certificates                                    \
      clang-15                                           \
      cmake                                              \
      curl                                               \
      git                                                \
      libomp-15-dev                                      \
      ninja-build                                        \
      openssh-client                                     \
      wget &&                                            \
      rm -rf /var/lib/apt/lists/*

# Set clang 15 tools
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100

# Install conda and dependencies
ENV PATH="/opt/conda/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda install -y python=3.12 cmake ninja && \
    conda install -y -c conda-forge mkl-static==2025.0.0 mkl-include==2025.0.0 onednn==3.5.3 mkl-devel==2025.0.0 && \
    conda install -y -c pytorch libpng libjpeg-turbo

# Install pytorch and libtorch
COPY pytorch-zconv.patch ${BUILD_DIR}/pytorch-zconv.patch
WORKDIR ${BUILD_DIR}
RUN git clone --recursive https://github.com/pytorch/pytorch.git pytorch && \
    cd pytorch && \
    git checkout v2.5.1 && git submodule sync && git submodule update --init --recursive && \
    git apply ${BUILD_DIR}/pytorch-zconv.patch && \
    python -mpip install -r requirements.txt && \
    export USE_CUDA=0 USE_ROCM=0 USE_XPU=0 _GLIBCXX_USE_CXX11_ABI=1 && \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" python setup.py install && \
    python tools/build_libtorch.py && \
    mkdir ${BUILD_DIR}/libtorch-install && \
    cp -r ${BUILD_DIR}/pytorch/torch/lib ${BUILD_DIR}/pytorch/torch/include ${BUILD_DIR}/pytorch/torch/share ${BUILD_DIR}/pytorch/torch/bin ${BUILD_DIR}/libtorch-install

# Install torch vision
WORKDIR ${BUILD_DIR}
RUN git clone --recursive https://github.com/pytorch/vision.git vision && \
    cd vision && \
    git checkout v0.20.1 && \
    python setup.py install

# Install google benchmark
WORKDIR ${BUILD_DIR}
RUN git clone https://github.com/google/benchmark.git benchmark && \
    cd benchmark && \
    git checkout v1.9.0 && \
    cmake -E make_directory "build" && \
    cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on \
                                 -DCMAKE_BUILD_TYPE=Release           \
                                 -DCMAKE_C_COMPILER=clang     \
                                 -DCMAKE_CXX_COMPILER=clang++ \
                                 ../ && \
    cmake --build "build" --config Release && \
    mkdir ${BUILD_DIR}/benchmark-install && \
    cmake --install build --config Release --prefix ${BUILD_DIR}/benchmark-install

# Install BLIS
WORKDIR ${BUILD_DIR}
RUN git clone https://github.com/flame/blis.git blis && \
    cd blis && \
    git checkout 1.0 && \
    mkdir ${BUILD_DIR}/blis-install && \
    ./configure --prefix="${BUILD_DIR}/blis-install" \
                --enable-threading=openmp            \
                --enable-cblas                       \
                CC=clang CXX=clang++                 \
                auto && \
    make install -j

# Install BLIS with Yaconv
COPY blis-yaconv.patch ${BUILD_DIR}/blis-yaconv.patch
WORKDIR ${BUILD_DIR}
RUN cp -r ${BUILD_DIR}/blis ${BUILD_DIR}/blis-yaconv && \
    cd blis-yaconv && \
    git apply ${BUILD_DIR}/blis-yaconv.patch && \
    mkdir ${BUILD_DIR}/blis-yaconv-install && \
    ./configure --prefix="${BUILD_DIR}/blis-yaconv-install" \
                --enable-threading=openmp            \
                --enable-cblas                       \
                -a yaconv                            \
                CC=clang CXX=clang++                 \
                auto && \
    make install -j

FROM ubuntu:22.04 AS test

ENV BUILD_DIR="/appbuild"

# Add conda to path
ENV PATH="/opt/conda/bin:${PATH}"

# Install essential packages and install vim sensible
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update && \
    apt-get install --assume-yes --no-install-recommends \
      build-essential                                    \
      ca-certificates                                    \
      clang-15                                           \
      cmake                                              \
      git                                                \
      libomp-15-dev                                      \
      linux-tools-common                                 \
      linux-tools-generic                                \
      linux-headers-generic                              \
      linux-headers-$(uname -r)                          \
      nano                                               \
      numactl                                            \
      unzip                                              \
      vim                                                \
      wget &&                                            \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p ~/.vim/pack/tpope/start && \
    cd ~/.vim/pack/tpope/start && \
    git clone https://tpope.io/vim/sensible.git

# Set clang 15 tools
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100

# Create and switch to non-root user and group
ARG user=artifact
ARG group=artifact
ARG uid=1000
ARG gid=1000
ENV HOME="/home/${user}"
RUN groupadd --gid ${gid} ${group} && \
    useradd --uid ${uid} -g ${group} --no-log-init -s /bin/bash -m ${user} && \
    mkdir ${HOME}/install

# Copy required files from build docker
COPY --from=builder /opt/conda /opt/conda
COPY --chown=${user}:${group} --from=builder ${BUILD_DIR}/benchmark-install ${HOME}/install/benchmark
COPY --chown=${user}:${group} --from=builder ${BUILD_DIR}/blis-install ${HOME}/install/blis
COPY --chown=${user}:${group} --from=builder ${BUILD_DIR}/blis-yaconv-install ${HOME}/install/blis-yaconv
COPY --chown=${user}:${group} --from=builder ${BUILD_DIR}/libtorch-install ${HOME}/install/libtorch

# Install python dependencies
RUN pip install scipy matplotlib pandas tabulate timm==1.0.13

# Install zconv single conv eval binaries
WORKDIR ${HOME}
COPY zero-copy-conv.zip ${HOME}/zero-copy-conv.zip
RUN unzip zero-copy-conv.zip -d zero-copy-conv && \
    rm zero-copy-conv.zip && \
    cd zero-copy-conv/single-conv && \
    mkdir build && cd build && \
    cmake -DCMAKE_C_COMPILER=clang                                 \
          -DCMAKE_CXX_COMPILER=clang++                             \
          -DBENCHMARK_INSTALL="${HOME}/install/benchmark"          \
          -DTORCH_INSTALL="${HOME}/install/libtorch"               \
          -DBLIS_INSTALL="${HOME}/install/blis"                    \
          -DBLIS_YACONV_INSTALL="${HOME}/install/blis-yaconv"      \
          -DUSE_MKL="ON"                                           \
          -DUSE_MKL_JIT="ON"                                       \
          -DBUILD_ZCONV_BLIS="ON"                                  \
          -DUSE_FIXED_ITERATIONS="OFF"                             \
          .. && \
    make -j && \
    mkdir "${HOME}/install/single-conv" && \
    cmake --install . --prefix "${HOME}/install/single-conv" && \
    cmake -DCMAKE_C_COMPILER=clang                                 \
          -DCMAKE_CXX_COMPILER=clang++                             \
          -DBENCHMARK_INSTALL="${HOME}/install/benchmark"          \
          -DTORCH_INSTALL="${HOME}/install/libtorch"               \
          -DBLIS_INSTALL="${HOME}/install/blis"                    \
          -DBLIS_YACONV_INSTALL="${HOME}/install/blis-yaconv"      \
          -DUSE_MKL="ON"                                           \
          -DUSE_MKL_JIT="ON"                                       \
          -DBUILD_ZCONV_BLIS="ON"                                  \
          -DUSE_FIXED_ITERATIONS="ON"                             \
          .. && \
    make -j && \
    mkdir "${HOME}/install/single-conv-fixed" && \
    cmake --install . --prefix "${HOME}/install/single-conv-fixed" && \
    cd .. && rm -r build

# Set user
USER ${uid}:${gid}

# Set home
WORKDIR ${HOME}

# Set default command
CMD ["bash"]

