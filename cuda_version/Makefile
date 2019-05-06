EXECUTABLE := cg
CU_FILES   := final_project.cu
CU_DEPS    :=
CC_FILES   := main.cpp

ARCH=$(shell uname | sed -e 's/-.*//g')
CUDA_PATH=$(TACC_CUDA_DIR)
OBJDIR=objs
INC_FLAGS=
CXX=g++
CXXFLAGS=-O3 -Wall $(INC_FLAGS)
LDFLAGS =

GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
GENCODE_FLAGS   := $(GENCODE_SM60) # For Pascal architecture
################################################################################
NVCC_LDFLAGS=-L$(CUDA_PATH)/lib64/ -lcudart -Wl,-rpath=$(CUDA_PATH)/lib64
NVCC=$(CUDA_PATH)/bin/nvcc -ccbin=$(CXX) 
NVCC_FLAGS=-O3 -m64 $(GENCODE_FLAGS) $(INC_FLAGS)

COMMON_OBJS=$(OBJDIR)/main.o

OBJS = $(COMMON_OBJS) $(OBJDIR)/final_project.o

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(NVCC_LDFLAGS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
