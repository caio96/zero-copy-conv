#!/usr/bin/env bash

function Help()
{
   echo "Run executables in build folder with the parameters defined in a csv file and generate an output log with performance or correctness metrics"
   echo
   echo "Syntax: $(basename "$0") BUILD_DIR CSV_FILE OUTPUT_FILE"
   echo "Options:"
   echo -e "\t--append-output: Allow output to be appended to existing file."
   echo -e "\t--batch-size [size]: Specify a custom batch size. Default is 1."
   echo -e "\t--repeats [repeats]: Specify the number of times each benchmark is repeated. Default is 1."
   echo -e "\t--threads [threads]: Specify the number of threads that each run can use (sets OMP_NUM_THREADS). Default is 1."
   echo -e "\t--core-range [range]: Specify the range of cores that a run may use. For example, to use only the first 12 cores, set it to '1-12'. Useful to select performance cores. Default is to use all cores."
   echo -e "\t--check-correctness: Run each benchmark once to check correctness. Ignores --repeats and --append-output."
   echo -e "\t-h: Print this Help."
   echo -e "\t-v: Verbose mode."
   echo
}

BATCH_SIZE="1"
REPEATS="1"
APPEND_OUTPUT="false"
CHECK_CORRECTNESS="false"
OMP_NUM_THREADS="1"
# Get list of cores from numactl, remove trailing spaces and get last number
CORE_RANGE="0-$(numactl --show | grep "physcpubind" | sed 's/[[:blank:]]*$//' | tr -s ' ' | rev | cut -d ' ' -f 1 | rev)"

# Parse Arguments
PARSED_ARGUMENTS=$(getopt -a -n "benchmark_runner" -o hv --long append-output,check-correctness,repeats:,batch-size:,threads:,core-range: -- "$@")
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
    --check-correctness)
      CHECK_CORRECTNESS="true"
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

# Create list with all executables found in build folder that start with "benchmark_"
executables=()
for executable in $(find "$BUILD_DIR" -type f -name "benchmark_*" | sort); do
  # if executable contains the word naive, skip it
  if [[ "$executable" == *"naive"* ]]; then
    continue
  fi
  executables+=("$executable")
done

CORRECTNESS_EXECUTABLE="$BUILD_DIR/correctness"
if [[ "$CHECK_CORRECTNESS" == "true" ]] && [[ ! -f "$CORRECTNESS_EXECUTABLE" ]]; then
  echo "Correctness executable not found. Maybe the name changed?"
  echo "Looking for $CORRECTNESS_EXECUTABLE"
  exit 1
fi

echo "Found ${#executables[@]} executables in $BUILD_DIR"
for executable in "${executables[@]}"; do
  echo "  - $(basename "$executable")"
done
echo ""

# Export google benchmark output format
export BENCHMARK_FORMAT="csv"

# Add header to output log if not appending
if [[ "$APPEND_OUTPUT" == "false" ]] && [[ "$CHECK_CORRECTNESS" == "false" ]]; then
  "${executables[0]}" 2> /dev/null | head -n1 >> "$OUTPUT_LOG"
fi

# Add header to correctness output
if [[ "$CHECK_CORRECTNESS" == "true" ]]; then
  $CORRECTNESS_EXECUTABLE | head -n1 >> "$OUTPUT_LOG"
fi

# Set number of threads
export OMP_NUM_THREADS

echo -e "Running with $REPEATS repetitions and $OMP_NUM_THREADS threads, using cores $CORE_RANGE\n"

# For each repetition
for repeat in $(seq "$REPEATS"); do
  echo "Starting repetition $repeat of $REPEATS --------------------"

  # For each configuration in the csv file
  while IFS=',' read -r conv_parameters occurrences models; do
    # Skip header
    if [[ "$conv_parameters" == "conv_parameters" ]]; then
      continue
    fi

    # Add custom batch size to parameters
    conv_parameters="$BATCH_SIZE $(echo "$conv_parameters" | cut -d ' ' -f2-)"

    echo "Running $conv_parameters"

    # Check correctness
    if [[ "$CHECK_CORRECTNESS" == "true" ]]; then
      "$CORRECTNESS_EXECUTABLE" ${conv_parameters} | tail -n +2 | tee -a "$OUTPUT_LOG"
      continue
    fi

    # For each executable (shuffled order)
    for executable in $(shuf -e "${executables[@]}"); do
      # Get random core list within range of size OMP_NUM_THREADS
      CORES=$(shuf -i "$CORE_RANGE" -n "$OMP_NUM_THREADS" | tr '\n' ',' | sed 's/,$//')
      # Run executable in a random core
      numactl --physcpubind "$CORES" "$executable" ${conv_parameters} 2> /dev/null | tail -n +2 | tee -a "$OUTPUT_LOG"
    done

  done < "$CSV_FILE"
done
