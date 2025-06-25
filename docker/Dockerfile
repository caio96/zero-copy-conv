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
WORKDIR ${BUILD_DIR}
RUN git clone --recursive https://github.com/caio96/pytorch-zero-copy.git pytorch && \
    cd pytorch && \
    git checkout v2.5.1-zero-copy && git submodule sync && git submodule update --init --recursive && \
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
WORKDIR ${BUILD_DIR}
RUN git clone https://github.com/caio96/blis-yaconv.git blis-yaconv && \
    cd blis-yaconv && \
    git checkout yaconv-update && \
    mkdir ${BUILD_DIR}/blis-yaconv-install && \
    ./configure --prefix="${BUILD_DIR}/blis-yaconv-install" \
                --enable-threading=openmp            \
                --enable-cblas                       \
                -a yaconv                            \
                CC=clang CXX=clang++                 \
                auto && \
    make install -j

# Install zconv single conv eval binaries
WORKDIR ${BUILD_DIR}
RUN git clone https://github.com/caio96/zero-copy-conv.git zconv-eval && \
    cd zconv-eval/single-conv && \
    mkdir build && cd build && \
    cmake -DCMAKE_C_COMPILER=clang                                 \
          -DCMAKE_CXX_COMPILER=clang++                             \
          -DBENCHMARK_INSTALL="${BUILD_DIR}/benchmark-install"     \
          -DTORCH_INSTALL="${BUILD_DIR}/libtorch-install"          \
          -DBLIS_INSTALL="${BUILD_DIR}/blis-install"               \
          -DBLIS_YACONV_INSTALL="${BUILD_DIR}/blis-yaconv-install" \
          -DUSE_MKL="ON"                                           \
          -DUSE_MKL_JIT="ON"                                       \
          -DBUILD_ZCONV_BLIS="ON"                                  \
          -DUSE_FIXED_ITERATIONS="OFF"                             \
          .. && \
    make -j && \
    mkdir ${BUILD_DIR}/zconv-eval-install && \
    cmake --install . --config Release --prefix ${BUILD_DIR}/zconv-eval-install

CMD ["bash"]

