#!/usr/bin/env bash

function Help()
{
   echo "Run executables in build folder with the parameters defined in a csv file and generate an output log with performance or correctness metrics"
   echo
   echo "Syntax: $(basename "$0") BUILD_DIR CSV_FILE OUTPUT_FILE"
   echo "Options:"
   echo -e "\t--append-output: Allow output to be appended to existing file."
   echo -e "\t--save-profile [profiler_output.txt]: Runs benchmarks with perf and saves perf output into separate output file."
   echo -e "\t--batch-size [size]: Specify a custom batch size. Default is 1."
   echo -e "\t--repeats [repeats]: Specify the number of times each benchmark is repeated. Default is 1."
   echo -e "\t--threads [threads]: Specify the number of threads that each run can use (sets OMP_NUM_THREADS). Default is 1."
   echo -e "\t--core-range [range]: Specify the range of cores that a run may use. For example, to use only the first 12 cores, set it to '1-12'. Useful to select performance cores. Default is to use all cores."
   echo -e "\t--rerun: Specify the csv file that contains 'conv_parameters' and 'rerun_methods', only these configurations will be run."
   echo -e "\t--check-correctness: Run each benchmark once to check correctness. Ignores --repeats and --append-output."
   echo -e "\t--parallel-single-thread-mode: Run each benchmark with a single thread but in parallel, using the configuration set with --threads and --core-range."
   echo -e "\t--run-zconv-blis: Enable running benchmark-zconv-blis if found, which is ignored by default."
   echo -e "\t-h: Print this Help."
   echo -e "\t-v: Verbose mode."
   echo
}

function show_progress {
    current="$1"
    total="$2"

    bar_size=$(( $(tput cols) - 23 - ${#total} - ${#current} ))
    bar_char_done="#"
    bar_char_todo="-"
    bar_percentage_scale=2

    # calculate the progress in percentage
    percent=$(bc <<< "scale=$bar_percentage_scale; 100 * $current / $total" )
    # The number of done and todo characters
    done=$(bc <<< "scale=0; $bar_size * $percent / 100" )
    todo=$(bc <<< "scale=0; $bar_size - $done" )

    # build the done and todo sub-bars
    done_sub_bar=$(printf "%${done}s" | tr " " "${bar_char_done}")
    todo_sub_bar=$(printf "%${todo}s" | tr " " "${bar_char_todo}")

    # output the bar
    printf "\rProgress: ${current}/${total} [${done_sub_bar}${todo_sub_bar}] %3.2f%%" "$percent"

    if [ "$total" -eq "$current" ]; then
        echo -e "\nDONE"
    fi
}

function CheckPerfParanoid()
{
  paranoid=$(cat '/proc/sys/kernel/perf_event_paranoid')
  if [ "$paranoid" -gt "1" ]; then
    echo "Please set kernel event paranoid to a least 1 to be able to use perf"
    echo "Use the following command:"
    echo "    sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'"
    exit 1
  fi
}

RUN_ZCONV_BLIS="false"
BATCH_SIZE="1"
REPEATS="1"
PERF_REPEATS="1"
PARALLEL_SINGLE_THREAD_MODE="false"
APPEND_OUTPUT="false"
CHECK_CORRECTNESS="false"
OMP_NUM_THREADS="1"
RERUN="false"
PROFILE="false"
PROFILE_OUTPUT=""
# Get list of cores from numactl, remove trailing spaces and get last number
CORE_RANGE="0-$(numactl --show | grep "physcpubind" | sed 's/[[:blank:]]*$//' | tr -s ' ' | rev | cut -d ' ' -f 1 | rev)"

# Parse Arguments
PARSED_ARGUMENTS=$(getopt -a -n "benchmark_runner" -o hv --long append-output,parallel-single-thread-mode,check-correctness,repeats:,batch-size:,threads:,core-range:,rerun,save-profile:,run-zconv-blis -- "$@")
if [ $? -ne 0 ]; then
  echo "Invalid option." >&2
  Help
  exit 1;
fi
eval set -- "$PARSED_ARGUMENTS"
while true; do
  case "$1" in
    -h)
      Help
      exit 0
      ;;
    -v)
      set -x
      shift
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --threads)
      OMP_NUM_THREADS="$2"
      shift 2
      ;;
    --append-output)
      APPEND_OUTPUT="true"
      shift
      ;;
    --core-range)
      CORE_RANGE="$2"
      shift 2
      ;;
    --rerun)
      RERUN="true"
      shift
      ;;
    --check-correctness)
      CHECK_CORRECTNESS="true"
      shift
      ;;
    --parallel-single-thread-mode)
      PARALLEL_SINGLE_THREAD_MODE="true"
      shift
      ;;
    --save-profile)
      PROFILE_OUTPUT="$2"
      PROFILE="true"
      shift 2
      ;;
    --run-zconv-blis)
      RUN_ZCONV_BLIS="true"
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unexpected option: $1"
      Help
      exit 1
      ;;
  esac
done

# Gets the first argument and checks if it is a directory.
BUILD_DIR=$1
if [[ ! -d $BUILD_DIR ]] || [[ -z $BUILD_DIR ]]; then
    echo "ERROR: Build path is empty or isn't a directory."
    Help
    exit 1
fi
BUILD_DIR=$(realpath "$BUILD_DIR")

CSV_FILE=$2
if [[ ! -f $CSV_FILE ]] || [[ -z $CSV_FILE ]]; then
    echo "ERROR: CSV file is empty or isn't a file."
    Help
    exit 1
fi
CSV_FILE=$(realpath "$CSV_FILE")

# Gets the second argument and checks if the file exists depeding of the append-output flag
OUTPUT_LOG=$3
if [[ "$APPEND_OUTPUT" == "true" ]] && [[ -z $OUTPUT_LOG ]]; then
    echo "ERROR: Output log path is empty."
    Help
    exit 1
elif [[ "$APPEND_OUTPUT" == "false" ]] && [[ -f $OUTPUT_LOG || -z $OUTPUT_LOG ]]; then
    echo "ERROR: Output log path is empty or file already exists. To append to an existing file use --append-output."
    Help
    exit 1
fi
OUTPUT_LOG=$(realpath "$OUTPUT_LOG")

# Checks for too many arguments
ERROR_ARGUMENT=$4
if [[ -n $ERROR_ARGUMENT ]]; then
  echo "ERROR: Too many arguments."
  Help
  exit 1
fi

if [[ $RERUN == "true" ]]; then
  header=$(head -n 1 "$CSV_FILE")
  if [[ $header != "conv_parameters,rerun_methods" ]]; then
    echo "ERROR: CSV file does not contain the expected header. This csv is generated by the summarize_results.py script."
    exit 1
  fi
fi

if [[ $PROFILE == "true" ]]; then
  if [[ "$APPEND_OUTPUT" == "true" ]] && [[ -z $PROFILE_OUTPUT ]]; then
    echo "ERROR: Profile output path is empty."
    Help
    exit 1
  elif [[ "$APPEND_OUTPUT" == "false" ]] && [[ -f $PROFILE_OUTPUT || -z $PROFILE_OUTPUT ]]; then
    echo "ERROR: Profile output path is empty or file already exists. To append to an existing file use --append-output."
    Help
    exit 1
  fi

  if [[ "$PARALLEL_SINGLE_THREAD_MODE" == "true" ]]; then
    echo "ERROR: Profiling in parallel single thread mode is not supported."
    exit 1
  fi

  CheckPerfParanoid
  PROFILE_OUTPUT=$(realpath "$PROFILE_OUTPUT")
  PERF_REPEATS="$REPEATS"
  REPEATS="1"
fi

# Create list with all executables found in build folder that start with "benchmark_"
executables=()
for executable in $(find "$BUILD_DIR" -type f -name "benchmark_*" | sort); do
  # If executable contains the word naive, skip it
  if [[ "$executable" =~ ^.*naive.*$ ]] ; then
    continue
  fi
  # Do not execute zconv_blis unless explicitly enabled
  if [[ "$executable" =~ ^.*blis.*$ ]]; then
    if [[ "$RUN_ZCONV_BLIS" == "false" ]]; then
        continue
    fi
  fi
  # Do not execute yaconv in multithreaded runs
  if [[ "$executable" =~ ^.*yaconv.*$ ]]; then
    if [[ "$PARALLEL_SINGLE_THREAD_MODE" == "false" ]] && [[ "$OMP_NUM_THREADS" != "1" ]]; then
        continue
    fi
  fi
  executables+=("$executable")
done

CORRECTNESS_EXECUTABLE="$BUILD_DIR/correctness"
if [[ "$CHECK_CORRECTNESS" == "true" ]] && [[ ! -f "$CORRECTNESS_EXECUTABLE" ]]; then
  echo "Correctness executable not found. Maybe the name changed?"
  echo "Looking for $CORRECTNESS_EXECUTABLE"
  exit 1
fi

if [[ "$CHECK_CORRECTNESS" == "false" ]]; then
  echo "Found ${#executables[@]} executables in $BUILD_DIR"
  for executable in "${executables[@]}"; do
    echo "  - $(basename "$executable")"
  done
  echo ""
fi

# Export google benchmark output format
export BENCHMARK_FORMAT="csv"

# Add header to output log if not appending
if [[ "$APPEND_OUTPUT" == "false" ]] && [[ "$CHECK_CORRECTNESS" == "false" ]]; then
  "${executables[0]}" 2> /dev/null | head -n1 >> "$OUTPUT_LOG"
fi

# Add header to correctness output
if [[ "$CHECK_CORRECTNESS" == "true" ]]; then
  $CORRECTNESS_EXECUTABLE 2> /dev/null | head -n1 >> "$OUTPUT_LOG"
fi

echo -e "Running with $REPEATS repetitions and $OMP_NUM_THREADS threads, using cores $CORE_RANGE\n"

# Run single threaded benchmarks in parallel
if [[ "$PARALLEL_SINGLE_THREAD_MODE" == "true" ]]; then
  echo "Single-threaded parallel mode"
  # Array to string, reconstructed inside parallel command
  export EXECUTABLES_STR="${executables[*]}"
  # Export variables so command inside parallel can see them
  export CHECK_CORRECTNESS BATCH_SIZE CORRECTNESS_EXECUTABLE OUTPUT_LOG RERUN BUILD_DIR
  parallel_log="$(dirname "$OUTPUT_LOG")/parallel_log.txt"

  # Parallel reads from the csv, each line is an input for the quoted commands
  numactl --physcpubind "$CORE_RANGE" parallel -a "$CSV_FILE" --skip-first-line --colsep "," --joblog "$parallel_log" --bar -j "$OMP_NUM_THREADS" '
      # Set single thread execution
      export OMP_NUM_THREADS=1

      # Get csv field
      conv_parameters={1}
      # Add custom batch size
      conv_parameters="$BATCH_SIZE $(echo "$conv_parameters" | cut -d " " -f2-)"

      # Check correctness
      if [[ "$CHECK_CORRECTNESS" == "true" ]]; then
        "$CORRECTNESS_EXECUTABLE" ${conv_parameters} | tail -n +2
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
          echo "[Error] When running with parameters: $conv_parameters"
        fi
      else
        if [[ "$RERUN" == "true" ]]; then
          rerun_methods={2}
          for method in $(echo "$rerun_methods" | tr "," "\n"); do
            # Choose executable based on method to rerun
            if [[ "$method" == "Naive" ]]; then
              executable="$BUILD_DIR/benchmark_naive"
            elif [[ "$method" == "Im2col" ]]; then
              executable="$BUILD_DIR/benchmark_im2col"
            elif [[ "$method" == "Yaconv" ]]; then
              executable="$BUILD_DIR/benchmark_yaconv"
            elif [[ "$method" =~ ^ZeroCopy.*$ ]]; then
              executable="$BUILD_DIR/benchmark_zero_copy"
            elif [[ "$method" =~ ^LibTorch_ZeroCopy2D.*$ ]]; then
              executable="$BUILD_DIR/benchmark_libtorch_zerocopy"
            elif [[ "$method" == "LibTorch" ]]; then
              executable="$BUILD_DIR/benchmark_libtorch"
            elif [[ "$method" == "OneDNN_any" ]]; then
              executable="$BUILD_DIR/benchmark_onednn_any"
            else
              echo "[Error] Unknown method: $method"
              continue
            fi

            if [[ ! -f "$executable" ]]; then
              echo "[Error] Executable not found: $executable"
              continue
            fi

            "$executable" ${conv_parameters} 2> /dev/null | tail -n +2
          done
        else
          # Recreate array from string
          IFS=" " read -r -a EXES <<< "$EXECUTABLES_STR"

          # Run each executable
          for executable in $(shuf -e "${EXES[@]}"); do
            "$executable" ${conv_parameters} 2> /dev/null | tail -n +2
          done
        fi

      fi
      ' ::: $(seq 1 $REPEATS) >> "$OUTPUT_LOG"

  exit 0
fi

# Set number of threads
export OMP_NUM_THREADS

total_iterations=$((REPEATS * ($(wc -l < "$CSV_FILE") - 1)))
current_iteration=0

# For each repetition
for repeat in $(seq "$REPEATS"); do
  # For each configuration in the csv file
  while IFS=',' read -r conv_parameters occurrences models; do
    # Show progress
    current_iteration=$((current_iteration + 1))
    show_progress "$current_iteration" "$total_iterations"

    # Add custom batch size to parameters
    conv_parameters="$BATCH_SIZE $(echo "$conv_parameters" | cut -d ' ' -f2-)"

    PERF_COMMAND=""
    if [[ "$PROFILE" == "true" ]]; then
      PERF_COMMAND="perf stat -x, -r${PERF_REPEATS} -e "page-faults,cpu_core/dTLB-loads/,cpu_core/dTLB-load-misses/,cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-stores/,cpu_core/L1-dcache-load-misses/,cpu_core/LLC-loads/,cpu_core/LLC-loads-misses/,cpu_core/LLC-stores/,cpu_core/LLC-store-misses/" -o $PROFILE_OUTPUT --append"
    fi

    # Check correctness
    if [[ "$CHECK_CORRECTNESS" == "true" ]]; then
      "$CORRECTNESS_EXECUTABLE" ${conv_parameters} | tail -n +2 >> "$OUTPUT_LOG"
      if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "Error running with parameters: $conv_parameters" | tee -a "${OUTPUT_LOG}.err"
      fi
      continue
    fi

    if [[ "$RERUN" == "true" ]]; then
      rerun_methods="$occurrences,$models"
      for method in $(echo "$rerun_methods" | tr -d '"' | tr "," "\n"); do
        # Choose executable based on method to rerun
        if [[ "$method" == "Naive" ]]; then
          executable="$BUILD_DIR/benchmark_naive"
        elif [[ "$method" == "Im2col" ]]; then
          executable="$BUILD_DIR/benchmark_im2col"
        elif [[ "$method" == "Yaconv" ]]; then
          executable="$BUILD_DIR/benchmark_yaconv"
        elif [[ "$method" =~ ^ZeroCopy.*$ ]]; then
          executable="$BUILD_DIR/benchmark_zero_copy"
        elif [[ "$method" =~ ^LibTorch_ZeroCopy2D.*$ ]]; then
          executable="$BUILD_DIR/benchmark_libtorch_zerocopy"
        elif [[ "$method" == "LibTorch" ]]; then
          executable="$BUILD_DIR/benchmark_libtorch"
        elif [[ "$method" == "OneDNN_any" ]]; then
          executable="$BUILD_DIR/benchmark_onednn_any"
        else
          echo "Unknown method: $method"
          continue
        fi

        if [[ ! -f "$executable" ]]; then
          echo "Executable not found: $executable"
          continue
        fi

        # head is needed when perf has repetition, causing the csv header to print multiple times
        numactl --physcpubind "$CORE_RANGE" $PERF_COMMAND "$executable" ${conv_parameters} 2> /dev/null | tail -n +2 | head -n 1 >> "$OUTPUT_LOG"
        if [[ "$PROFILE" == "true" ]]; then
          # append run information to profile output
          tail -n 1 "$OUTPUT_LOG" >> "$PROFILE_OUTPUT"
        fi
      done
    else
      # For each executable (shuffled order)
      for executable in $(shuf -e "${executables[@]}"); do
        numactl --physcpubind "$CORE_RANGE" $PERF_COMMAND "$executable" ${conv_parameters} 2> /dev/null | tail -n +2 | head -n 1 >> "$OUTPUT_LOG"
        if [[ "$PROFILE" == "true" ]]; then
          # append run information to profile output
          tail -n 1 "$OUTPUT_LOG" >> "$PROFILE_OUTPUT"
        fi
      done
    fi

  done < <(tail -n +2 "$CSV_FILE")
done
