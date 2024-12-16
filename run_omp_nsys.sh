#!/bin/bash

source params.sh

# Check for --profile flag
PROFILE=false
for arg in "$@"; do
    if [[ $arg == "--profile" ]]; then
        PROFILE=true
    fi
done

# Create necessary directories
if [ ! -d "$OMP_OUT_FOLDER" ]; then
    mkdir "$OMP_OUT_FOLDER"
fi
if [ ! -d "$MEASURES_FOLDER" ]; then
    mkdir "$MEASURES_FOLDER"
fi

# Create a directory for profiling data
PROFILE_FOLDER="$MEASURES_FOLDER/profiles"
if [ ! -d $PROFILE_FOLDER ]; then
    mkdir $PROFILE_FOLDER
fi

# Check if model and dataset files are available
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: missing model $MODEL_PATH!"
    exit 1
fi

if [ ! -d "$DTASET_FOLDER" ]; then
    echo "Error: missing dataset $DTASET_FOLDER!"
    exit 1
fi

# Check if executable is available
if [ ! -f "omp_bin/vit.exe" ]; then
    echo "Error: missing omp_bin/vit.exe file!"
    echo "Run compile.sh script"
    exit 1
fi

# Create the output CSV file if it does not exist
if [ ! -f "$MEASURES_FOLDER/omp_$NUM_THREADS.csv" ]; then
    touch "$MEASURES_FOLDER/omp_$NUM_THREADS.csv"
fi
echo "batch_size;model_depth;load_cvit_time;load_cpic_time;foreward_time[];store_cprd_time" >"$MEASURES_FOLDER/omp_$NUM_THREADS.csv"

# Set the number of OpenMP threads
export OMP_NUM_THREADS=$NUM_THREADS

# Run the program with or without nsys profiling
if [ "$PROFILE" = true ]; then
    echo "Running with nsys profiling..."
    
    # Create a unique timestamp for each run
    TIMESTAMP=$(date +%Y%m%d%H%M%S)

    # Run the program with nsys profiling
    for ((i=0; i<$DTASET_DIM; i++)); do
        nsys profile -o "$PROFILE_FOLDER/omp_profile_${TIMESTAMP}_$NUM_THREADS$i" \
            ./omp_bin/vit.exe "$MODEL_PATH" "$DTASET_FOLDER/pic_$i.cpic" \
            "$OMP_OUT_FOLDER/prd_$i.cprd" "$MEASURES_FOLDER/omp_$NUM_THREADS.csv"
    done

else
    # Run the program normally without profiling
    for ((i=0; i<$DTASET_DIM; i++)); do
        ./omp_bin/vit.exe "$MODEL_PATH" "$DTASET_FOLDER/pic_$i.cpic" \
            "$OMP_OUT_FOLDER/prd_$i.cprd" "$MEASURES_FOLDER/omp_$NUM_THREADS.csv"
    done
fi

echo "vit executed with $NUM_THREADS threads"

# Analyze the time measures
python3 scripts/analyze_time_measures.py "$MEASURES_FOLDER/omp_$NUM_THREADS.csv" \
    "$MEASURES_FOLDER/omp_$NUM_THREADS""_summary.txt"

