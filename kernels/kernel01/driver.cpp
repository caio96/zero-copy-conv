#include "kernel.h"
#include <benchmark/benchmark.h>

extern "C" void conv_2d(float *__restrict__ input, float *__restrict__ output,
                        float *__restrict__ filter, int N, int H, int W, int C,
                        int FH, int FW, int M, int PH, int PW);

static void BM_MAIN(benchmark::State &state) {

  const int N = 1;
  const int H = 56;
  const int W = 56;
  const int C = 64;
  const int FH = 3;
  const int FW = 3;
  const int M = 256;
  const int PH = 1;
  const int PW = 1;
  const int OH = H + 2 * PH - FH + 1;
  const int OW = W + 2 * PW - FW + 1;

  float *input;
  float *filter;
  float *output;

  input = (float *)malloc(N * H * W * C * sizeof(float));
  filter = (float *)malloc(FH * FW * C * M * sizeof(float));
  output = (float *)malloc(N * OH * OW * M * sizeof(float));

  if (input == NULL || filter == NULL || output == NULL) {
    fprintf(stderr, "Some error in malloc!\n");
    exit(-1);
  }

  // Run kernel
  for (auto _ : state) {
    conv_2d(input, output, filter, N, H, W, C, FH, FW, M, PH, PW);
  }

  free(input);
  free(filter);
  free(output);

  return;
}

BENCHMARK(BM_MAIN)
    ->Name("conv-2d")
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->ReportAggregatesOnly(true);

// Run the benchmark
BENCHMARK_MAIN();
