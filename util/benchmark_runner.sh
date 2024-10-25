#!/bin/bash
# Run executables and generate a log with execution times

# Gets the location of the script
function getScriptLocation {
    SOURCE="${BASH_SOURCE[0]}"
    while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
        DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
        SOURCE="$(readlink "$SOURCE")"
        [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
    done
        DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
        echo "$DIR"
}

function Help()
{
   echo "Run executables in build folder with the parameters defined in a csv file and generate an output log with performance metrics"
   echo
   echo "Syntax: $(basename "$0") BUILD_DIR CSV_FILE OUTPUT_FILE"
   echo "Options:"
   echo -e "\t--append-output: Allow output to be appended to existing file."
   echo -e "\t--batch-size [size]: Specify a custom batch size. Default is 1."
   echo -e "\t--repeats [repeats]: Specify the number of times each benchmark is repeated. Default is 1."
   echo -e "\t-h: Print this Help."
   echo -e "\t-v: Verbose mode."
   echo
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

BATCH_SIZE="1"
REPEATS="1"
APPEND_OUTPUT="false"
# Parse Arguments
PARSED_ARGUMENTS=$(getopt -a -n "benchmark_runner" -o hv --long append-output,repeats:,batch-size: -- "$@")
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
    --append-output)
      APPEND_OUTPUT="true"
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

# Check if perf will work
CheckPerfParanoid

# Create list with all executables found in build folder that start with "benchmark_"
executables=()
for executable in $(find "$BUILD_DIR" -type f -name "benchmark_*" | sort); do
  # if executable contains the word naive, skip it
  if [[ "$executable" == *"naive"* ]]; then
    continue
  fi
  executables+=("$executable")
done

echo "Found ${#executables[@]} executables in $BUILD_DIR"
for executable in "${executables[@]}"; do
  echo "  - $(basename "$executable")"
done
echo ""

# Export google benchmark output format
export BENCHMARK_FORMAT="csv"

# Add header to output log if not appending
if [[ "$APPEND_OUTPUT" == "false" ]]; then
  "${executables[0]}" 2> /dev/null | head -n1 | tee -a "$OUTPUT_LOG"
fi

# For each repetition
for repeat in $(seq "$REPEATS"); do
  echo "Starting repetition $repeat of $REPEATS --------------------"

  # For each configuration in the csv file
  while IFS=',' read -r id batch ic ih iw oc oh ow fh hw pt pb pl pr sh sw dh dw gr; do
    # Skip header
    if [[ "$id" == "ID" ]]; then
      continue
    fi

    echo ""
    echo "Running benchmark for ID: $id"
    echo "  - Batch: $BATCH_SIZE"
    echo "  - Input: $ic x $ih x $iw"
    echo "  - Output: $oc x $oh x $ow"
    echo "  - Filter: $fh x $hw"
    echo "  - Padding: $pt $pb $pl $pr"
    echo "  - Stride: $sh $sw"
    echo "  - Dilation: $dh $dw"
    echo "  - Groups: $gr"

    # For each executable (shuffled order)
    for executable in $(shuf -e "${executables[@]}"); do
      # Get random number
      CORE_NUM=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')
      # Run executable in a random core
      numactl -C $(( $RANDOM % $CORE_NUM )) "$executable" "$BATCH_SIZE" "$ic" "$ih" "$iw" "$oc" "$oh" "$ow" "$fh" "$hw" "$pt" "$pb" "$pl" "$pr" "$sh" "$sw" "$dh" "$dw" "$gr" 2> /dev/null | tail -n1 | tee -a "$OUTPUT_LOG"
    done

  done < "$CSV_FILE"
done

