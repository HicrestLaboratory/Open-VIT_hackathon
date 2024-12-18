# Dataset and model parameters
export DTASET_FOLDER="data_0" # Path of the dataset folder
export DTASET_DIM="1" # How many batches to use in the selected dataset
export MODEL_PATH="model/vit.cvit" # Path of the model file

# Outputs parameters
export CPP_OUT_FOLDER="out" # Folder where cpp predictions will be stored
export OMP_OUT_FOLDER="omp_out" # Folder where omp predictions will be stored
export ACC_OUT_FOLDER="acc_out" # Folder where omp predictions will be stored
export MEASURES_FOLDER="measures" # Folder where time measures will be stored

# Prediction comparison parameters
export CPRD_HIGH_THRESHOLD="0.0001" # needed in scripts/compare_cpred.py
export CPRD_LOW_THRESHOLD="0.000001" # needed in scripts/compare_cpred.py

# OMP threads parameter
export NUM_THREADS="16"
