#include <mma.h>
#include <stdio.h>
#include "utils.cu"

#define N 1024
#define TILE_SIZE 16
#define WARP_SIZE 32

#define IN_DTYPE unsigned char
#define ACC_DTYPE int

__global__ void matmul(IN_DTYPE *a, IN_DTYPE *b, ACC_DTYPE *c)
{
    int block_i = blockIdx.y; // block index along row (y) axis
    int block_j = blockIdx.x; // block index along col (x) axis

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, IN_DTYPE, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, IN_DTYPE, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, ACC_DTYPE> acc_frag;

    nvcuda::wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
    for (int wmma_block_index = 0; wmma_block_index < N / TILE_SIZE; wmma_block_index++)
    {
        nvcuda::wmma::load_matrix_sync(a_frag, a + (block_i * TILE_SIZE * N) + (wmma_block_index * TILE_SIZE), N);
        nvcuda::wmma::load_matrix_sync(b_frag, b + (wmma_block_index * TILE_SIZE * N) + (block_j * TILE_SIZE), N);

        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    nvcuda::wmma::store_matrix_sync(c + (block_i * TILE_SIZE * N) + (block_j * TILE_SIZE), acc_frag, N, nvcuda::wmma::mem_row_major);
}

int main()
{
    srand(time(NULL));

    ACC_DTYPE *a = (ACC_DTYPE *)malloc(N * N * sizeof(ACC_DTYPE));
    ACC_DTYPE *b = (ACC_DTYPE *)malloc(N * N * sizeof(ACC_DTYPE));
    ACC_DTYPE *c = (ACC_DTYPE *)malloc(N * N * sizeof(ACC_DTYPE));

    // fill a & b
    // matrix_random_fp16valued(a, N * N);
    // matrix_random_fp16valued(b, N * N);

    IN_DTYPE *a_h = (IN_DTYPE *)malloc(N * N * sizeof(IN_DTYPE));
    IN_DTYPE *b_h = (IN_DTYPE *)malloc(N * N * sizeof(IN_DTYPE));

    // for (int i = 0; i < N * N; i++)
    // {
    //     a_h[i] = __float2half(a[i]);
    //     b_h[i] = __float2half(b[i]);
    // }

    IN_DTYPE *d_a, *d_b;
    ACC_DTYPE *d_c;
    cudaMalloc(&d_a, N * N * sizeof(IN_DTYPE));
    cudaMalloc(&d_b, N * N * sizeof(IN_DTYPE));
    cudaMalloc(&d_c, N * N * sizeof(ACC_DTYPE));
    cudaMemcpy(d_a, a_h, N * N * sizeof(IN_DTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_h, N * N * sizeof(IN_DTYPE), cudaMemcpyHostToDevice);

    dim3 grid_dim(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(N, TILE_SIZE));
    dim3 block_dim(WARP_SIZE);
    printf("LAUNCHING with grid_dim: (%d, %d) and block_dim: %d\n", grid_dim.x, grid_dim.y, block_dim.x);

    uint64_t start = nanos();
    matmul<<<grid_dim, block_dim>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    uint64_t end = nanos();

    cudaMemcpy(c, d_c, N * N * sizeof(ACC_DTYPE), cudaMemcpyDeviceToHost);

    double gflop = (2.0 * N * N * N) * 1e-9;
    double s = (end - start) * 1e-9;
    printf("%f GFLOP/S -- %.2f ms\n", gflop / s, s * 1e3);

    // {
    //     // compute naive reference matmul on cpu
    //     printf("Computing reference matmul result on cpu\n");
    //     float *reference_c = (float *)malloc(N * N * sizeof(float));
    //     matmul_c(a, b, reference_c, N);

    //     // check each item
    //     printf("Comparing reference result with gpu result\n");
    //     matrix_eq(reference_c, c, N);
    //     printf("ALL GOOD\n");
    //     free(reference_c);
    // }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a_h);
    free(b_h);
    free(a);
    free(b);
    free(c);
}