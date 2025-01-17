// very heavily modified https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtFp8Matmul/sample_cublasLt_LtFp8Matmul.cu
#include <time.h>
#include <stdio.h>
#include <cublasLt.h>
#include "helpers.h"

#define ITERS 1000
#define N 4096
#define TILE_SIZE 16
#define WARP_SIZE 32

uint64_t nanos()
{
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main()
{
    srand(time(NULL));

    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    int m = N;
    int k = N;
    int n = N;

    unsigned char *A, *B;
    int *C;
    cudaMalloc(&A, N * N * sizeof(unsigned char));
    cudaMalloc(&B, N * N * sizeof(unsigned char));
    cudaMalloc(&C, N * N * sizeof(int));

    size_t workspaceSize = 1024 * 1024 * 4;
    void *workspace;
    cudaMalloc(&workspace, workspaceSize);

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasOperation_t transb = CUBLAS_OP_T;
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, k));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, n));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, n));

    cublasLtMatmulPreference_t preference = NULL;
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0)
    {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    // Scaling factors
    int32_t alpha = 1;
    int32_t beta = 0;

    uint64_t start = nanos();

    for (int i = 0; i < ITERS; i++)
    {
        checkCublasStatus(cublasLtMatmul(ltHandle,
                                         operationDesc,
                                         &alpha,
                                         A,
                                         Adesc,
                                         B,
                                         Bdesc,
                                         &beta,
                                         C,
                                         Cdesc,
                                         C,
                                         Cdesc,
                                         &heuristicResult.algo,
                                         workspace,
                                         workspaceSize,
                                         0));
    }

    cudaDeviceSynchronize();
    uint64_t end = nanos();

    double gflop = (2.0 * N * N * N) * 1e-9 * (float)ITERS;
    double s = (end - start) * 1e-9;
    printf("%f TOPS -- %.2f ms\n", (gflop / 1000.) / s, s * 1e3);

    checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(workspace);
}
