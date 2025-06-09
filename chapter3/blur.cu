#include <stdio.h>
#include <cuda.h>
#include "cuda_helper.h"

static int bx, by;
static int repetitions;
static int blur_size;

__device__ unsigned char blurPixel(unsigned char *pin, int col, int row, int width, int height, int blur_size)
{
    int pixVal = 0;
    int pixels = 0;

    for (int rowOffset = -blur_size; rowOffset <= blur_size; ++rowOffset)
    {
        for (int colOffset = -blur_size; colOffset <= blur_size; ++colOffset)
        {
            int curRow = row + rowOffset;
            int curCol = col + colOffset;
            if (curCol >= 0 && curCol < width && curRow >= 0 && curRow < height)
            {
                pixVal += pin[curRow * width + curCol];
                ++pixels;
            }
        }
    }

    return (unsigned char) (pixVal / pixels);
}

__global__ void blurKernel(unsigned char *pout, unsigned char *pin, int width, int height, int blur_size)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height)
    {
        pout[row * width + col] = blurPixel(pin, col, row, width, height, blur_size);
    }
}

void blur(unsigned char *pout, unsigned char *pin, int width, int height)
{
    unsigned char *pout_d, *pin_d;
    size_t size = (size_t)width * height * sizeof(unsigned char);
    cudaMalloc((void **)&pout_d, size);
    cudaMalloc((void **)&pin_d, size);
    cudaMemcpy(pin_d, pin, size, cudaMemcpyHostToDevice);

    int gx = (width + bx - 1) / bx;
    int gy = (height + by - 1) / by;
    printf("Grid dimension: %d x %d\n", gx, gy);
    printf("Block dimension: %d x %d\n", bx, by);
    dim3 dimGrid(gx, gy, 1);
    dim3 dimBlock(bx, by, 1);

    // Warmup
    blurKernel<<<dimGrid, dimBlock>>>(pout_d, pin_d, width, height, blur_size);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Benchmark
    double start_time = cpuSecond();
    for (int i = 0; i < repetitions; ++i)
    {
        blurKernel<<<dimGrid, dimBlock>>>(pout_d, pin_d, width, height, blur_size);
        CHECK(cudaGetLastError());
    }
    cudaDeviceSynchronize();
    double end_time = cpuSecond();
    double elapsed_time = (end_time - start_time) * 1e6 / repetitions;
    printf("Processed time: %.3lf us\n", elapsed_time);

    cudaMemcpy(pout, pout_d, size, cudaMemcpyDeviceToHost);
    cudaFree(pout_d);
    cudaFree(pin_d);
}

void skip_comments(FILE *fp)
{
    int ch;
    while ((ch = fgetc(fp)) == '#')
    {
        while (fgetc(fp) != '\n')
            ; // skip to end of comment line
    }
    ungetc(ch, fp);
}

FILE *read_pgm_header(const char *filename, int *width, int *height)
{
    FILE *fp = fopen(filename, "rb");
    char format[3];
    int maxval;
    fscanf(fp, "%s", format);
    skip_comments(fp);
    fscanf(fp, "%d", width);
    skip_comments(fp);
    fscanf(fp, "%d", height);
    skip_comments(fp);
    fscanf(fp, "%d", &maxval);
    fgetc(fp);
    return fp;
}

void read_pgm_image(FILE *fp, unsigned char *img, int width, int height)
{
    fread(img, 1, width * height, fp);
    fclose(fp);
}

void write_pgm_image(const char *filename, const unsigned char *gray, int width, int height)
{
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(gray, 1, width * height, fp);
    fclose(fp);
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "%s <blur_size> <bx> <by> <r>\n", argv[0]);
        exit(1);
    }
    blur_size = atoi(argv[1]);
    bx = atoi(argv[2]);
    by = atoi(argv[3]);
    repetitions = atoi(argv[4]);

    int width, height;
    unsigned char *image, *blurred;
    size_t size;

    // Read image
    FILE *fp = read_pgm_header("sample_gray.pgm", &width, &height);
    size = (size_t)width * height * sizeof(unsigned char);
    image = (unsigned char *)malloc(size);
    read_pgm_image(fp, image, width, height);
    printf("Image size: %d x %d\n", width, height);

    // Process image
    blurred = (unsigned char *) malloc(size);
    blur(blurred, image, width, height);

    // Write image
    write_pgm_image("sample_blur.pgm", blurred, width, height);

    free(image);
    free(blurred);

    return 0;
}