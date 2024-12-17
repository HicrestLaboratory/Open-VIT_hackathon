# C++ Vision Transformer

## Project Structure

This project contains an implementation of all the necessary components as well as a full ViT model. In particular:
- Library headers (`include/`)
- A single-threaded serial implementation (`src/`)
- An OpenMP parallel implementation (`omp_src/`)
- Some utility scripts (`scripts/`)
- A pre-defined model to be used for weight initialization (`model/`)
- Some datasets (`data_0/` `data_1/` `data_2/` `data_3/`)
- Files for individual component testing (`test_src/`)
- Scripts for easy benchmarking (`params.sh` `run_cpp.sh` `run_omp.sh` `elaborate.sh`)

## --
# Things to do when offload the code on Leonardo system 

## Load the NVHPC compiler, compile your code with: 
```
module load nvhpc/24.3
make -f makefile.acc
```

## After compiling code: run and analysis with nsight system

## To run nsys profile:
```
- Serial: ./run_cpp_nsys.sh --profile 
- Openmp: ./run_omp_nsys.sh --profile
- Openacc./run_acc_nsys.sh --profile
```

## Download NVIDIA Nsight Systems on your preferred laptop:

[Click here](https://developer.nvidia.com/nsight-systems/get-started)


## --


## How to Compile

C++ single-threaded serial implementation:
```
make bin/vit.exe
```

OpenMP parallel implementation:
```
make omp_bin/vit.exe
```

Clean the folder from all compiled file:
```
make clean
```

## How to Run Locally

C++ single-threaded serial implementation:
```
./bin/vit.exe <model.cvit> <in_file.cpic> <out_file.cprd> <measures_file.csv>
```

OpenMP parallel implementation:
```
export OMP_NUM_THREADS=<num_threads>
./omp_bin/vit.exe <model.cvit> <in_file.cpic> <out_file.cprd> <measures_file.csv>
```

Example:
```
./bin/vit.exe model/vit.cvit data_0/pic_0.cpic out/prd_0.cprd measures/cpp.csv
export OMP_NUM_THREADS=16
./omp_bin/vit.exe model/vit.cvit data_0/pic_0.cpic omp_out/prd_0.cprd measures/omp_16.csv
```

You can use two scripts to easily automate this process (they both rely on `params.sh`):
```
# Edit params.sh
bash run_cpp.sh
bash run_omp.sh
```

# Run Nsys profile 
```
./run_cpp_nsys.sh --profile
```


## How to Run on Leonardo

TODO

## Performance Analysis

When the model run, they automatically create output labes as well as measure files.

If you used `bash run_cpp.sh` you will already find a `cpp_summary.txt` file in the measure folder, same for `bash run_omp.sh`.

Otherwise you can manually compute it with:
```
python3 scripts/analyze_time_measures.py <measure_file.csv> <out_file.txt>
```

## Output Comparison

Compare two prediction files:
```
python3 scripts/compare_cpred.py <prediction_a.cprd> <prediction_b.cprd> <out_file.txt> <high_treshold> <low_treshold>
```

These comparisons have a section for each batch, so for large datasets you may want to summarize them in a shorter file:
```
python3 scripts/summary_cpred_comparison.py <comparison_file.txt> <summarized_file.txt>
```

If you used `bash run_cpp.sh` and `bash run_cpp.sh`, you can run another script to automatically generate the comparisons and summarize them
in the `out_comparison/` folder:
```
# It relies as well on params.sh
bash elaborate.sh
```

# Other Useful Scripts

Plot an input batch file:
```
python3 scripts/plot_cpic.py <batch_file.cpic>
```

Plot an output prediction file:
```
python3 scripts/plot_prediction.py <prediction_file.cprd>
```

Create an input batch of a random quantity pictures, each one composed by random pixels:
```
python3 scripts/random_cpic <out_file.cpic> <min_batch_dim> <max_batch_dim> <num_channels> <picture_height> <picture_width> <min_pixel_val> <max_pixel_val>
```

Generate the lines of code to insert a random value tensor in your test file:
```
python3 scripts/create_tensor.py <dim_0> <optional_dim_1> <optional_dim_2> <optional_dim_3>
```

## Unit Tests

To run an individual component test, first modify the proper file in `test_src/test_<module_name>.cpp`.

Where `<module_name>` can be: `datatypes`, `modules`, `mlp`, `conv2d`, `attention`, `block`, `patch_embed`, `vision_transformer` and `utils`.

Then compile it:
```
make test_bin/test_<module_name>.exe
```
And run it:
```
test_bin/test_<module_name>.exe
```
