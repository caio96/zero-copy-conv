# Yaconv Update

Usage:

- Clone, build, and install the updated BLIS repo with Yaconv's implementation

```sh
git clone git@github.com:caio96/blis-conv.git
cd blis-conv
git checkout yaconv-update
./configure --prefix=/path/to/blis-install -a yaconv auto
make install -j
```

- Clone, build, and install the Google Benchmark framework

```sh
git clone https://github.com/google/benchmark.git
cd benchmark
git checkout v1.9.0
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../
cmake --build "build" --config Release
cmake --install build --config Release --prefix /path/to/benchmark-install
```

- Update paths in CMakeLists.txt to the installation of google benchmark and blis

- Build

```sh
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
make
```

- To run benchmarks, run the executables `benchmark_im2col`, `benchmark_naive`, `benchmark_yaconv`, and `correctness`
