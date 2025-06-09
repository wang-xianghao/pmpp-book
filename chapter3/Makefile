NVCC = nvcc
NVCC_FLAGS = -O3 -I../include

SOURCES = grayScale.cu blur.cu
TARGET = grayScale blur

.PHONY: all
all: $(TARGET)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lm

.PHONY: clean
clean:
	rm -f $(TARGET)