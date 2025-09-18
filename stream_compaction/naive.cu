#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;

        // Block variables
        int blockSize = 128;
        dim3 threadsPerBlock(blockSize);

        // Data buffers to swap between each itertion
        int* dev_dataBuf1;
        int* dev_dataBuf2;

        void setBlockSize(int newBlockSize) {
          blockSize = newBlockSize;
          threadsPerBlock = dim3(blockSize);
        }

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        
        __global__ void kernNaiveScan(int n, int currOffset, int *odata, const int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index >= currOffset) {
                odata[index] = idata[index - currOffset] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int iters = ceil(log2(n));
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // set up device arrays
            cudaMalloc((void**)&dev_dataBuf1, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_dataBuf1 failed!");
            cudaMalloc((void**)&dev_dataBuf2, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_dataBuf2 failed!");
            cudaMemcpy(dev_dataBuf1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed!");

            timer().startGpuTimer();

            for (int d = 1; d <= iters; d++) {
                // calculate power of 2 offset with bitshift
                int currOffset = 1 << d - 1;
                kernNaiveScan<<<fullBlocksPerGrid, threadsPerBlock>>> (n, currOffset, dev_dataBuf2, dev_dataBuf1);
                int* temp = dev_dataBuf1;
                dev_dataBuf1 = dev_dataBuf2;
                dev_dataBuf2 = temp;
            }

            timer().endGpuTimer();

            // Copy to host
            if (n > 0) {
              odata[0] = 0;
            }
            // dev_dataBuf1 is inclusive scan, so shift to make exclusive
            cudaMemcpy(odata + 1, dev_dataBuf1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to host failed!");

            // Free device arrays
            cudaFree(dev_dataBuf1);
            checkCUDAError("cudaFree dev_dataBuf1 failed!");
            cudaFree(dev_dataBuf2);
            checkCUDAError("cudaFree dev_dataBuf2 failed!");

        }
    }
}
