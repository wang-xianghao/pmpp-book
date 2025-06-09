#include <stdio.h>
#include <cuda.h>
#include "cuda_helper.h"

const int CHANNELS = 3;
static int bx, by;
static int repetitions;

__global__ void colorGrayscaleConvertionKernel(unsigned char *pout,
                                               unsigned char *pin,
                                               int width, int height)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * CHANNELS;
        unsigned int r = pin[rgbOffset];
        unsigned int g = pin[rgbOffset + 1];
        unsigned int b = pin[rgbOffset + 2];
        pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

void colorGrayscaleConvertion(unsigned char *pout,
                              unsigned char *pin,
                              int width, int height)
{
    unsigned char *pout_d, *pin_d;
    size_t graySize = (size_t) width * height * sizeof(unsigned char); 
    size_t rgbSize = graySize * CHANNELS;
    cudaMalloc((void **) &pout_d, graySize);
    cudaMalloc((void **) &pin_d, rgbSize);
    cudaMemcpy(pin_d, pin, rgbSize, cudaMemcpyHostToDevice);

    int gx = (width + bx - 1) / bx;
    int gy = (height + by - 1) / by;
    printf("Grid dimension: %d x %d\n", gx, gy);
    printf("Block dimension: %d x %d\n", bx, by);
    dim3 dimGrid(gx, gy, 1);
    dim3 dimBlock(bx, by, 1);
    
    // Warmup
    colorGrayscaleConvertionKernel<<<dimGrid, dimBlock>>>(pout_d, pin_d, width, height);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    
    // Benchmark
    double start_time = cpuSecond();
    for (int i = 0; i < repetitions; ++ i)
    {
        colorGrayscaleConvertionKernel<<<dimGrid, dimBlock>>>(pout_d, pin_d, width, height);
        CHECK(cudaGetLastError());
    }
    cudaDeviceSynchronize();
    double end_time = cpuSecond();
    double elapsed_time = (end_time - start_time) * 1e6 / repetitions;
    printf("Processed time: %.3lf us\n", elapsed_time);

    cudaMemcpy(pout, pout_d, graySize, cudaMemcpyDeviceToHost);
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

FILE *read_ppm_header(const char *filename, int *width, int *height)
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

void read_ppm_image(FILE *fp, unsigned char *img, int width, int height)
{
    fread(img, 1, width * height * 3, fp);
    fclose(fp);
}

void write_pgm_image(const char *filename, const unsigned char *gray, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(gray, 1, width * height, fp);
    fclose(fp);
}

int main(int argc, char *argv[])
{
    if (argc != 4) 
    {
        fprintf(stderr, "%s <bx> <by> <r>\n", argv[0]);
        exit(1);
    }
    bx = atoi(argv[1]);
    by = atoi(argv[2]);
    repetitions = atoi(argv[3]);
    
    int width, height;
    unsigned char *image, *grayImage;

    // Read image
    FILE *fp = read_ppm_header("images/sample.ppm", &width, &height);
    image = (unsigned char *) malloc(width * height * CHANNELS * sizeof(unsigned char));
    read_ppm_image(fp, image, width, height);
    printf("Image size: %d x %d\n", width, height);

    // Process image
    grayImage = (unsigned char *) malloc(width * height * sizeof(unsigned char));
    colorGrayscaleConvertion(grayImage, image, width, height);

    // Write image
    write_pgm_image("sample_gray.pgm", grayImage, width, height);

    free(image);
    free(grayImage);

    return 0;
}