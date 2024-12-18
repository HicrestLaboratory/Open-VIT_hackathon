#!/bin/bash

source params.sh

# Set CUDA and NVHPC environment variables
# Set CUDA and NVHPC environment variables
NVHPC_INCLUDE_PATH=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3
CUDA_INCLUDE_PATH=$NVHPC_INCLUDE_PATH/cuda/12.3/include
CUDA_LIB_PATH=$NVHPC_INCLUDE_PATH/cuda/12.3/targets/x86_64-linux/lib

export LD_LIBRARY_PATH=$CUDA_LIB_PATH:$LD_LIBRARY_PATH

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH


# Check if profiling is enabled
PROFILE=false
for arg in "$@"; do
    if [[ $arg == "--profile" ]]; then
        PROFILE=true
    fi
done

# Create necessary directories
if [ ! -d $CPP_OUT_FOLDER ]; then
    mkdir $CPP_OUT_FOLDER
fi
if [ ! -d $MEASURES_FOLDER ]; then
    mkdir $MEASURES_FOLDER
fi
if [ ! -d $ACC_OUT_FOLDER ]; then
    mkdir $ACC_OUT_FOLDER
fi

# Create a directory for profiling data
PROFILE_FOLDER="$MEASURES_FOLDER/profiles"
if [ ! -d $PROFILE_FOLDER ]; then
    mkdir $PROFILE_FOLDER
fi

# Check if model and dataset files are available
if [ ! -f $MODEL_PATH ]; then
    echo Error: missing model $MODEL_PATH!
    exit 1
fi

if [ ! -d $DTASET_FOLDER ]; then
    echo Error: missing dataset $DTASET_FOLDER!
    exit 1
fi

# Check if executable is available
if [ ! -f "acc_bin/vit.exe" ]; then
    echo Error: missing acc_bin/vit.exe file!
    echo Run compile.sh script
    exit 1
fi

# Create the CSV file if it doesn't exist
if [ ! -f $MEASURES_FOLDER/acc.csv ]; then
    touch $MEASURES_FOLDER/acc.csv
fi

# Initialize CSV with headers
echo "batch_size;model_depth;load_cvit_time;load_cpic_time;foreward_time[];store_cprd_time" >$MEASURES_FOLDER/acc.csv

# Function to run the executable and capture GPU information
run_executable() {
    local i=$1
    local output
    output=$(./acc_bin/vit.exe $MODEL_PATH $DTASET_FOLDER/pic_$i.cpic $ACC_OUT_FOLDER/prd_$i.cprd $MEASURES_FOLDER/acc.csv 2>&1)
    echo "$output"
    if echo "$output" | grep -q "Actually I am using GPU"; then
        echo "GPU has been used for batch $i"
    else
        echo "GPU has not been used for batch $i"
    fi
}

# Run the profiling loop if --profile flag is set
if $PROFILE; then
    echo "Running with nsys profiling..."
    
    # Create a unique timestamp for each run
    TIMESTAMP=$(date +%Y%m%d%H%M%S)
    
    # Loop over dataset
    for ((i=0; i<$DTASET_DIM; i++)); do
        # Run the executable with nsys profiling, store the output in the PROFILE_FOLDER with a unique name
        nsys profile --output=$PROFILE_FOLDER/profile_${TIMESTAMP}_$i.nsys-rep ./acc_bin/vit.exe $MODEL_PATH $DTASET_FOLDER/pic_$i.cpic $ACC_OUT_FOLDER/prd_$i.cprd $MEASURES_FOLDER/acc.csv
    done
else
    # Without profiling, just run the normal execution
    for ((i=0; i<$DTASET_DIM; i++)); do
        run_executable $i
    done
fi

echo vit executed

# Run Python script to analyze the time measures
python3 scripts/analyze_time_measures.py $MEASURES_FOLDER/acc.csv $MEASURES_FOLDER/acc_summary.txt
