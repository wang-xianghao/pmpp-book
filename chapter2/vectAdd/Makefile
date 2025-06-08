NVCC = nvcc
NVCC_FLAGS = -O3 -G -I../../include

SOURCES = vectAdd.cu
TARGET = vectAdd

.PHONY: all
all: $(TARGET)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lm

.PHONY: clean
clean:
	rm -f $(TARGET)