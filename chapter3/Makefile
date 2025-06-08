NVCC = nvcc
NVCC_FLAGS = -O3 -I../include

SOURCES = grayScale.cu
TARGET = grayScale

.PHONY: all
all: $(TARGET)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lm

.PHONY: clean
clean:
	rm -f $(TARGET)