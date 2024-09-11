#!/usr/bin/env bash
# Compiles a kernel

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
  echo "Compiles a kernel"
  echo
  echo "Syntax: $(basename "$0") KERNEL_PATH OUTPUT_PATH"
  echo "Options:"
  echo -e "\t-h: Print this Help."
  echo -e "\t-v: Verbose mode."
  echo
}

function sourceConfigFile() {
  # Gets this script path.
  scriptPath=$(getScriptLocation)
  configFilePath="$scriptPath/../config.file"
  # Checks the config.file
  if [ ! -f "$configFilePath" ]; then
    echo "Please create config.file"
    Help
    exit 2
  fi
  source "$configFilePath"
}

function checkDependencies() {
  if [[ -z $CLANG || ! -f $CLANG ]]; then
    echo "$CLANG"
    echo "clang not found!"
    exit 1
  fi

  if [[ -z $CLANGPP || ! -f $CLANGPP ]]; then
    echo "$CLANGPP"
    echo "clang++ not found!"
    exit 1
  fi
}

function checkClangVersion() {
  if [[ -z $CLANG_VERSION ]]; then
    echo "Clang version not defined!"
    exit 1
  fi

  # Check clang version
  $CLANG --version | grep -q --fixed-strings "$CLANG_VERSION"
  if [[ $? -ne 0 ]]; then
    echo "Incorrect clang version, please set it up according to the config.file"
    echo "Corrent version is $CLANG_VERSION"
    exit 1
  fi
}

# Parse Arguments
PARSED_ARGUMENTS=$(getopt -a -n "compile-benchmark" -o hv -- "$@")
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
KERNEL_DIR=$1
if [[ -z $KERNEL_DIR ]] || [[ ! -d $KERNEL_DIR ]]; then
  echo "ERROR: Kernel path is empty or isn't a directory."
  echo "       path: $KERNEL_DIR"
  Help
  exit 1
fi
KERNEL_DIR=$(realpath "$KERNEL_DIR")

# Gets the second argument and checks if it is a directory.
OUTPUT_DIR=$2
if [[ -z $OUTPUT_DIR ]] || [[ ! -d $OUTPUT_DIR ]]; then
  echo "ERROR: Output path is empty or isn't a directory."
  Help
  exit 1
fi
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

# Checks for too many arguments
ERROR_ARGUMENT=$3
if [[ -n $ERROR_ARGUMENT ]]; then
  echo "ERROR: Too many arguments."
  Help
  exit 1
fi

# Source config and check dependencies and check clang version
sourceConfigFile
checkDependencies
checkClangVersion

# These flags are needed if google benchmark is not global
if [ -n "$GOOGLE_BENCHMARK_INSTALL" ]; then
  GOOGLE_BENCHMARK_FLAGS="-isystem $GOOGLE_BENCHMARK_INSTALL/include -L$GOOGLE_BENCHMARK_INSTALL/lib"
else
  GOOGLE_BENCHMARK_FLAGS=""
fi

# Kernel files
KERNEL="$KERNEL_DIR/kernel.c"
KERNEL_OPT="$KERNEL_DIR/kernel-opt.c"
DRIVER="$KERNEL_DIR/driver.cpp"

# Check if kernel files exist
if [[ ! -f "$DRIVER" ]] || [[ ! -f "$KERNEL" ]] || [[ ! -f "$KERNEL_OPT" ]]; then
  echo "One of these was not found:"
  echo -e "\t-${KERNEL}"
  echo -e "\t-${KERNEL_OPT}"
  echo -e "\t-${DRIVER}"
  exit 1
fi

# Use absolute path
KERNEL=$(realpath "$KERNEL")
KERNEL_OPT=$(realpath "$KERNEL_OPT")
DRIVER=$(realpath "$DRIVER")

cd $OUTPUT_DIR

# Compile base version of kernel function
$CLANG -c "$KERNEL"       \
       -Wall -O3          \
       -march=native      \
       -S -emit-llvm      \
       -o "$OUTPUT_DIR"/"kernel".ll
if [ $? != 0 ]; then
  echo "Compiling kernel failed"
  exit 1
fi

# Compile base version of kernel function
$CLANG -c "$KERNEL"       \
       -Wall -O3          \
       -march=native      \
       -S                 \
       -o "$OUTPUT_DIR"/"kernel".s
if [ $? != 0 ]; then
  echo "Compiling kernel failed"
  exit 1
fi

# Compile and link kernel with google benchmark main function
$CLANGPP "$OUTPUT_DIR"/"kernel".ll "$DRIVER" \
         -std=c++11                \
         -Wall -O3                 \
         -march=native             \
         -fuse-ld="${LLVM_LLD}"    \
         ${GOOGLE_BENCHMARK_FLAGS} -lm -lbenchmark -lpthread \
         -o "$OUTPUT_DIR"/"kernel".exe

# Compile opt version of kernel function
$CLANG -c "${KERNEL_OPT}" \
       -Wall -O3          \
       -march=native      \
       -S -emit-llvm      \
       -o "$OUTPUT_DIR"/"kernel-opt".ll
if [ $? != 0 ]; then
  echo "Compiling kernel failed"
  exit 1
fi

# Compile opt version of kernel function
$CLANG -c "${KERNEL_OPT}" \
       -Wall -O3          \
       -march=native      \
       -S                 \
       -o "$OUTPUT_DIR"/"kernel-opt".s
if [ $? != 0 ]; then
  echo "Compiling kernel failed"
  exit 1
fi

# Compile and link kernel with google benchmark main function
$CLANGPP "$OUTPUT_DIR"/"kernel-opt".ll "$DRIVER" \
         -std=c++11                \
         -Wall -O3                 \
         -march=native             \
         -fuse-ld="${LLVM_LLD}"    \
         ${GOOGLE_BENCHMARK_FLAGS} -lm -lbenchmark -lpthread \
         -o "$OUTPUT_DIR"/"kernel-opt".exe
