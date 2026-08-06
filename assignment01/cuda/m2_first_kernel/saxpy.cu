/* code for prob 2.9
不允许 include common.h。错误检查宏和 cudaEvent 计时都要自己写一遍。
命令行用法：./saxpy <n>，n 是元素个数。

输入数据按固定公式生成（都是 float）：
x[i] = ((i % 2048) - 1024) * 0.5f，
y[i] = (i % 1024) - 512

kernel 算完把 y 拷回 host，用 double 累加所有 y[i]，输出一行 SUM=<总和>（用 printf("SUM=%.0f\n", s) 这样的格式，
同一行里可以再带上 n 和 kernel 毫秒数），SUM 结果将用于对拍检验程序正确性，exit code 应为 0。
n=0 时输出 SUM=0，exit code 为 0（0 个 block 的 kernel launch 是非法的，特判即可）。
判测脚本覆盖 
n∈{0,1,31,1024,1025,2^20,2^20+3}
*/

#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 包住每个 CUDA API 调用，出错立刻报出文件、行号和原因。
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                    cudaGetErrorName(err_), __FILE__, __LINE__,           \
                    cudaGetErrorString(err_));                            \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

__global__ void saxpy(const float* x, float* y, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 2^20 exceed the maximum thread number. grid stride needed
    for (int i=idx; i < n; i = i + blockDim.x * gridDim.x){
        y[i] = (float) 2 * x[i] + y[i];
    }
}


int main(int argc, char* argv[]){
    int n = std::stoi(argv[1]);
    std::cout << "接收到元素数量 n = " << n << std::endl;

    // special case: n == 0
    if (n==0) {
        printf("SUM=%.0f\n", 0);
        return 0;
    }

    size_t bytes = (size_t)n * sizeof(float);
    float* y = nullptr;
    float* x = nullptr;

    // Unified moemory style
    CUDA_CHECK(cudaMallocManaged(&y, bytes));
    CUDA_CHECK(cudaMallocManaged(&x, bytes));

    // x, y init
    for (int i=0; i<n ;i = i+1){
        x[i] = ((i % 2048) - 1024) * 0.5f;
        y[i] = (i % 1024) - 512;
    }

    saxpy<<<1024, 256>>>(x, y, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // calculate SUM
    double SUM = 0;
    for (int i=0; i<n; i = i+1){
        SUM = SUM + y[i];
    }
    printf("SUM=%.0f\n", SUM);
    return 0;
}