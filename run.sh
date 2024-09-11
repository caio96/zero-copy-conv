#!/usr/bin/env bash
# Compiles all kernels, executes all kernels and generates output graphs

# Gets the location of the script
function getScriptLocation {
  SOURCE="${BASH_SOURCE[0]}"
  while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
    DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
  done
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  echo $DIR
}

function Help()
{
  echo "Compiles all kernels"
  echo "Then executes all kernels and generates output graphs"
  echo
  echo "Syntax: $(basename "$0") OUTPUT_PATH"
  echo "Options:"
  echo -e "\t-h: Print this Help."
  echo -e "\t-v: Verbose mode."
  echo -e "\t--repetitions [XX]: Number of repetitions when executing the kernels. Default is 3."
  echo -e "\t--no-recompile: Skip compiling, execute and generate summary."
  echo -e "\t--no-execution: Skip executing and generating graphs."
  echo -e "\t--no-summary: Skip printing summary."
  echo -e "\t-v: Verbose mode."
  echo
}

function sourceConfigFile() {
  # Gets this script path.
  scriptPath=$(getScriptLocation)
  configFilePath="$scriptPath/config.file"
  # Checks the config.file
  if [ ! -f "$configFilePath" ]; then
    echo "Please create config.file"
    Help
    exit 2
  fi
  source "$configFilePath"
}

function checkDependencies() {
  if [[ -z $REPO_ROOT || ! -d $REPO_ROOT ]]; then
    echo "$REPO_ROOT"
    echo "scripts not found!"
    exit 1
  fi
}

SUMMARY="true"
EXECUTION="true"
COMPILE="true"
TIME_GRAPHS=""
REPETITIONS="3"
# Parse Arguments
PARSED_ARGUMENTS=$(getopt -a -n "run" -o hv --long no-recompile,no-execution,no-summary,repetitions: -- "$@")
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
    --no-recompile)
      COMPILE="false"
      shift
      ;;
    --no-execution)
      EXECUTION="false"
      shift
      ;;
    --no-summary)
      SUMMARY="true"
      shift
      ;;
    --repetitions)
      REPETITIONS="$2"
      shift 2
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
OUTPUT_DIR=$1
if [[ -z $OUTPUT_DIR ]] || [[ ! -d $OUTPUT_DIR ]]; then
  echo "ERROR: Output path is empty or isn't a directory."
  Help
  exit 1
fi
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

# Checks for too many arguments
ERROR_ARGUMENT=$2
if [[ -n $ERROR_ARGUMENT ]]; then
  echo "ERROR: Too many arguments."
  Help
  exit 1
fi

# Source config and check dependencies and check clang version
sourceConfigFile
checkDependencies

# Get kernels dir
KERNELS_DIR="${REPO_ROOT}/kernels"
if [[ ! -d $KERNELS_DIR ]]; then
  echo "$KERNELS_DIR"
  echo "ERROR: Kernel path not found."
  Help
  exit 1
fi
KERNELS_DIR=$(realpath "$KERNELS_DIR")

COMPILE_KERNEL_SH="${REPO_ROOT}/scripts/compile-kernel.sh"
if [[ ! -f $COMPILE_KERNEL_SH ]]; then
  echo "$COMPILE_KERNEL_SH"
  echo "ERROR: compile-kernel.sh not found."
  Help
  exit 1
fi
COMPILE_KERNEL_SH=$(realpath "$COMPILE_KERNEL_SH")

OUTPUT_SUMMARY_PY="${REPO_ROOT}/scripts/output-summary.py"
if [[ ! -f $OUTPUT_SUMMARY_PY ]]; then
  echo "$OUTPUT_SUMMARY_PY"
  echo "ERROR: output-summary.py not found."
  Help
  exit 1
fi
OUTPUT_SUMMARY_PY=$(realpath "$OUTPUT_SUMMARY_PY")

# Find every kernel and compile it
for kernel in $(find "${KERNELS_DIR}" -mindepth 1 -name "kernel*" -type d | sort); do
  kernel_name=$(basename "$kernel")

  output="${OUTPUT_DIR}/${kernel_name}"
  if [[ -d "$output" && "$COMPILE" == "false" ]] ; then
    echo "Skipping compiling $kernel_name"
    continue
  else
    echo "Compiling $kernel_name"
  fi

  mkdir -p "$output"

  $COMPILE_KERNEL_SH "$kernel" "$output"
done

if [[ "$EXECUTION" == "true" ]]; then

  # Go to every compiled kernel and execute it
  for kernel in $(find "${OUTPUT_DIR}" -mindepth 1 -name "kernel*" -type d | shuf); do
    if [[ -f "$kernel/kernel.exe" ]] || [[ -f "$kernel/kernel-opt.exe" ]] ; then
      echo "Executing $(basename "$kernel")"
      perf stat -e "{mem_load_retired.l1_hit,frontend_retired.l1i_miss,mem_load_retired.l1_miss},{mem_load_retired.l2_hit,mem_load_retired.l2_miss},{mem_load_retired.l3_hit,mem_load_retired.l3_miss}" "$kernel"/kernel.exe --benchmark_repetitions="$REPETITIONS" &> "$kernel/kernel.txt"
      perf stat -e "{mem_load_retired.l1_hit,frontend_retired.l1i_miss,mem_load_retired.l1_miss},{mem_load_retired.l2_hit,mem_load_retired.l2_miss},{mem_load_retired.l3_hit,mem_load_retired.l3_miss}" "$kernel"/kernel-opt.exe --benchmark_repetitions="$REPETITIONS" &> "$kernel/kernel-opt.txt"
    else
      echo "Kernel executable not found in $kernel"
    fi
  done

  # Generate summary
  if [[ "$SUMMARY" == "true" ]]; then
      echo ""
      echo ""
      $OUTPUT_SUMMARY_PY "${OUTPUT_DIR}"
  fi

fi

