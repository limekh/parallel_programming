#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>

using namespace std;
using Matrix1D = vector<float>;

// Проверка ошибок CUDA
inline void check_cuda(cudaError_t code, const char* filename, int lineno) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%d) at %s:%d\n",
                cudaGetErrorString(code), code, filename, lineno);
        exit(code);
    }
}
#define CUDA_CHECK(call) check_cuda(call, __FILE__, __LINE__)

// Генерация матрицы
Matrix1D generate_sq_matrix(int n, float minValue, float maxValue) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(minValue, maxValue);
    
    Matrix1D matrix(n * n);
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = dist(gen);
    }
    return matrix;
}

// CUDA ядро
__global__ void mul_sq_matrix_kernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Оптимизированное ядро с разделяемой памятью
__global__ void mul_sq_matrix_shared_kernel(const float* A, const float* B, float* C, int n) {
    const int TILE_SIZE = 16;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + tx;
        if (row < n && aCol < n)
            As[ty][tx] = A[row * n + aCol];
        else
            As[ty][tx] = 0.0f;
        
        int bRow = t * TILE_SIZE + ty;
        if (bRow < n && col < n)
            Bs[ty][tx] = B[bRow * n + col];
        else
            Bs[ty][tx] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];
        
        __syncthreads();
    }
    
    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 4) {
            cerr << "Usage: " << argv[0] 
                 << " N blockX blockY [minValue maxValue] [use_shared]" << endl;
            return 1;
        }
        
        int n = stoi(argv[1]);
        int blockX = stoi(argv[2]);
        int blockY = stoi(argv[3]);
        float minValue = (argc >= 5) ? stof(argv[4]) : 0.0f;
        float maxValue = (argc >= 6) ? stof(argv[5]) : 10.0f;
        bool useShared = (argc >= 7) ? (stoi(argv[6]) != 0) : false;
        
        // Проверки для GTX 1060 3GB
        if (n > 2000) {
            cerr << "Warning: Matrix size " << n << " may exceed 3GB VRAM!" << endl;
        }
        
        Matrix1D matrix_a = generate_sq_matrix(n, minValue, maxValue);
        Matrix1D matrix_b = generate_sq_matrix(n, minValue, maxValue);
        Matrix1D result_matrix(n * n, 0.0f);
        
        size_t bytes = n * n * sizeof(float);
        long long operations_counts = 2LL * n * n * n;
        
        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));
        
        CUDA_CHECK(cudaMemcpy(d_A, matrix_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, matrix_b.data(), bytes, cudaMemcpyHostToDevice));
        
        dim3 block(blockX, blockY);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Прогрев
        if (useShared) {
            mul_sq_matrix_shared_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
        } else {
            mul_sq_matrix_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Замер
        CUDA_CHECK(cudaEventRecord(start));
        if (useShared) {
            mul_sq_matrix_shared_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
        } else {
            mul_sq_matrix_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        CUDA_CHECK(cudaMemcpy(result_matrix.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        
        cout << "Matrix size: " << n << "x" << n << endl;
        cout << "Block config: (" << blockX << ", " << blockY << ")" << endl;
        cout << "Memory type: " << (useShared ? "shared" : "global") << endl;
        cout << "Kernel execution time: " << milliseconds << " ms" << endl;
        cout << "Operations: " << operations_counts << endl;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        
        return 0;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}