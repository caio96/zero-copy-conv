#!/usr/bin/env python3

import argparse
import glob
import re
import sys
from math import sqrt
from pathlib import Path

import scipy.stats as st
import tabulate


def parse_file(input):
    mean = -1
    stddev = -1
    iterations = -1
    parsing_perf = False
    counters = {}

    for line in open(input, "r"):
        line = line.strip()
        line = re.sub(" +", " ", line)
        if len(line) == 0:
            continue

        # Parse google benchmark output
        if "mean" in line:
            mean = float(line.split(" ")[1])
            iterations_value = int(line.split(" ")[5])
            if iterations == -1:
                iterations = iterations_value
            else:
                assert iterations == iterations_value
            continue

        if "stddev" in line:
            stddev = float(line.split(" ")[1])
            continue

        # Beginning of perf output
        if "Performance counter stats for" in line:
            parsing_perf = True
            continue

        # End of perf output
        if "seconds time elapsed" in line:
            parsing_perf = False
            continue

        if "<not supported>" in line or "<not counted>" in line:
            continue

        # Parse perf output
        if parsing_perf:
            counter = line.split(" ")[0]
            # Skip comment lines
            if counter == "#":
                continue
            counter = float(re.sub(",", "", counter))
            counter_name = line.split(" ")[1]
            counters[counter_name] = counter

    conf_low, conf_high = st.norm.interval(
        confidence=0.95, loc=mean, scale=stddev / sqrt(iterations)
    )
    conf = (conf_high - conf_low) / 2

    return (mean, conf, iterations, counters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Google Benchmark output logs and generate summary."
    )

    parser.add_argument(
        "input_dir",
        help="Directory containing the Google Benchmark output files."
        "Files must be named kernel.txt and kernel-opt.txt,"
        "and these files must be in a folder named kernelXX"
        "(XX is the number of the kernel).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print("Input directory is not a directory of does not exist", file=sys.stderr)
        sys.exit(1)
    input_dir = input_dir.absolute()

    benchmark_results = []
    perf_results = []

    for kernel_dir in sorted(input_dir.glob("kernel*/")):

        input_base = Path(kernel_dir / "kernel.txt")
        input_opt = Path(kernel_dir / "kernel-opt.txt")

        if not input_base.exists() or not input_base.is_file():
            print(
                "{} file is not a file of does not exist".format(input_base),
                file=sys.stderr,
            )
            continue

        if not input_opt.exists() or not input_opt.is_file():
            print(
                "{} file is not a file of does not exist".format(input_opt),
                file=sys.stderr,
            )
            continue

        input_base = input_base.absolute()
        input_opt = input_opt.absolute()

        base_mean, base_conf, base_iters, base_counters = parse_file(input_base)
        opt_mean, opt_conf, opt_iters, opt_counters = parse_file(input_opt)
        assert base_iters == opt_iters

        percent_speedup = (base_mean - opt_mean) / opt_mean * 100

        benchmark_results.append(
            (
                Path(kernel_dir).name,
                "{:.2f}".format(base_mean),
                "{:.2f}".format(base_conf),
                "{:.2f}".format(opt_mean),
                "{:.2f}".format(opt_conf),
                "{:.0f}%".format(percent_speedup),
                base_iters,
            )
        )

        perf_row = [Path(kernel_dir).name]
        counter_name_row = []
        for counter_name in base_counters.keys():
            if opt_counters[counter_name] == 0:
                continue

            percent_improvement = (base_counters[counter_name] - opt_counters[counter_name]) / opt_counters[counter_name] * 100
            perf_row.append("{:.0f}%".format(percent_improvement))
            counter_name_row.append(counter_name)
        perf_results.append(perf_row)


    header = [
        "Kernel",
        "Base mean",
        "Base conf",
        "Opt mean",
        "Opt conf",
        "Speedup",
        "Iters",
    ]
    print(tabulate.tabulate(benchmark_results, headers=header))
    print("\n\n")

    header = ["Kernel"]
    header.extend(counter_name_row)
    header = [cell.split('.')[-1] for cell in header]
    print(tabulate.tabulate(perf_results, headers=header))
