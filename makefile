# Set the NVHPC compiler instead of g++
CC := nvc++

# Add OpenACC flag for compilation and execution, and use correct OpenMP flag (-mp)
CFLAGS := -std=c++11 -O3 -Minfo=mp,acc
OMPFLAGS := -mp -acc

# Bin, Obj, and Src Folders
BIN_FOLDER := bin
OBJ_FOLDER := bin/obj
SRC_FOLDER := src

OMP_BIN_FOLDER := omp_bin
OMP_OBJ_FOLDER := omp_bin/obj
OMP_SRC_FOLDER := omp_src

TEST_BIN_FOLDER := test_bin
TEST_OBJ_FOLDER := test_bin/obj
TEST_SRC_FOLDER := test_src

# Compilation Targets
all : vit

clean :
	rm -rf ./$(OBJ_FOLDER) ./$(BIN_FOLDER) ./$(OMP_OBJ_FOLDER) ./$(OMP_BIN_FOLDER) \
		   ./$(TEST_OBJ_FOLDER) ./$(TEST_BIN_FOLDER) \
		   ./out_comparison ./test_files

# OBJs
$(OBJ_FOLDER)/datatypes.o \
$(OBJ_FOLDER)/modules.o \
$(OBJ_FOLDER)/mlp.o \
$(OBJ_FOLDER)/conv2d.o \
$(OBJ_FOLDER)/attention.o \
$(OBJ_FOLDER)/block.o \
$(OBJ_FOLDER)/patch_embed.o \
$(OBJ_FOLDER)/vision_transformer.o \
$(OBJ_FOLDER)/utils.o \
$(OBJ_FOLDER)/main.o \
\
: $(OBJ_FOLDER)/%.o : $(SRC_FOLDER)/%.cpp
	mkdir -p $(OBJ_FOLDER)
	$(CC) -c $(CFLAGS) $^ -o $@

# Executables
$(BIN_FOLDER)/vit.exe : \
\
$(OBJ_FOLDER)/datatypes.o \
$(OBJ_FOLDER)/modules.o \
$(OBJ_FOLDER)/mlp.o \
$(OBJ_FOLDER)/conv2d.o \
$(OBJ_FOLDER)/attention.o \
$(OBJ_FOLDER)/block.o \
$(OBJ_FOLDER)/patch_embed.o \
$(OBJ_FOLDER)/vision_transformer.o \
$(OBJ_FOLDER)/utils.o \
$(OBJ_FOLDER)/main.o
	mkdir -p $(BIN_FOLDER)
	$(CC) $(CFLAGS) $^ -o $@

# OMP OBJs with OpenACC
$(OMP_OBJ_FOLDER)/datatypes.o \
$(OMP_OBJ_FOLDER)/modules.o \
$(OMP_OBJ_FOLDER)/conv2d.o \
$(OMP_OBJ_FOLDER)/attention.o \
$(OMP_OBJ_FOLDER)/vision_transformer.o \
\
: $(OMP_OBJ_FOLDER)/%.o : $(OMP_SRC_FOLDER)/%.cpp
	mkdir -p $(OMP_OBJ_FOLDER)
	$(CC) -c $(CFLAGS) $(OMPFLAGS) $^ -o $@

# OMP Executables with OpenACC
$(OMP_BIN_FOLDER)/vit.exe : \
\
$(OMP_OBJ_FOLDER)/datatypes.o \
$(OMP_OBJ_FOLDER)/modules.o \
$(OBJ_FOLDER)/mlp.o \
$(OMP_OBJ_FOLDER)/conv2d.o \
$(OMP_OBJ_FOLDER)/attention.o \
$(OBJ_FOLDER)/block.o \
$(OBJ_FOLDER)/patch_embed.o \
$(OMP_OBJ_FOLDER)/vision_transformer.o \
$(OBJ_FOLDER)/utils.o \
$(OBJ_FOLDER)/main.o
	mkdir -p $(OMP_BIN_FOLDER)
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@

# Test OBJs
$(TEST_OBJ_FOLDER)/test_datatypes.o \
$(TEST_OBJ_FOLDER)/test_modules.o \
$(TEST_OBJ_FOLDER)/test_mlp.o \
$(TEST_OBJ_FOLDER)/test_conv2d.o \
$(TEST_OBJ_FOLDER)/test_attention.o \
$(TEST_OBJ_FOLDER)/test_block.o \
$(TEST_OBJ_FOLDER)/test_patch_embed.o \
$(TEST_OBJ_FOLDER)/test_vision_transformer.o \
$(TEST_OBJ_FOLDER)/test_utils.o \
\
: $(TEST_OBJ_FOLDER)/%.o : $(TEST_SRC_FOLDER)/%.cpp
	mkdir -p $(TEST_OBJ_FOLDER)
	$(CC) -c $(CFLAGS) $^ -o $@

# Test Executables
$(TEST_BIN_FOLDER)/test_datatypes.exe \
$(TEST_BIN_FOLDER)/test_modules.exe \
$(TEST_BIN_FOLDER)/test_mlp.exe \
$(TEST_BIN_FOLDER)/test_conv2d.exe \
$(TEST_BIN_FOLDER)/test_attention.exe \
$(TEST_BIN_FOLDER)/test_block.exe \
$(TEST_BIN_FOLDER)/test_patch_embed.exe \
$(TEST_BIN_FOLDER)/test_vision_transformer.exe \
$(TEST_BIN_FOLDER)/test_utils.exe \
\
: $(TEST_BIN_FOLDER)/%.exe : \
\
$(OBJ_FOLDER)/datatypes.o \
$(OBJ_FOLDER)/modules.o \
$(OBJ_FOLDER)/mlp.o \
$(OBJ_FOLDER)/conv2d.o \
$(OBJ_FOLDER)/attention.o \
$(OBJ_FOLDER)/block.o \
$(OBJ_FOLDER)/patch_embed.o \
$(OBJ_FOLDER)/vision_transformer.o \
$(OBJ_FOLDER)/utils.o \
$(TEST_OBJ_FOLDER)/%.o
	mkdir -p $(TEST_BIN_FOLDER)
	$(CC) $(CFLAGS) $^ -o $@

