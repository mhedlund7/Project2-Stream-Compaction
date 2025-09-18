#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Block variables
        #define blockSize 128
        dim3 threadsPerBlock(blockSize);

        // Data buffers
        int* dev_idata;
        int* dev_bools;
        int* dev_negbools;
        int* dev_scanned;
        int* dev_writingIndices;
        int* dev_indices;
        int* dev_scatterBools;
        int* dev_odata;

        __global__ void kernUpSweep(int nearestPow2, int currOffset, int* data) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          int parentIdx = (index + 1) * currOffset * 2 - 1;
          int leftChildIdx = parentIdx - currOffset;
          if (parentIdx >= nearestPow2) {
            return;
          }
          data[parentIdx] += data[leftChildIdx];
        }

        __global__ void kernDownSweep(int nearestPow2, int currOffset, int* data) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          int parentIdx = (index + 1) * currOffset * 2 - 1;
          int leftChildIdx = parentIdx - currOffset;
          if (parentIdx >= nearestPow2) {
            return;
          }
          int temp = data[leftChildIdx];
          data[leftChildIdx] = data[parentIdx];
          data[parentIdx] += temp;
        }

        __global__ void kernMapToBits(int n, int bit, const int* idata, int* bools) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          // isolate the desired bit
          bools[index] = (idata[index] >> bit) & 1;
        }

        __global__ void kernNegateArray(int n, const int* bools, int* negBools) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          negBools[index] = 1 - bools[index];
        }

        __global__ void kernGetWritingIndices(int n, const int* scanned, int* writingIndices, int totalFalses) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          writingIndices[index] = index - scanned[index] + totalFalses;
        }

        __global__ void kernGetScatterIndices(int n, const int* bools, const int* writingIndices, const int* scanned, int* indices) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          indices[index] = bools[index] ? writingIndices[index] : scanned[index];
        }


        void radix(int n, int* odata, const int* idata) {
          int iters = ilog2ceil(n);
          int nearestPow2 = 1 << ilog2ceil(n);
          // set up device arrays
          cudaMalloc((void**)&dev_idata, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_idata failed!");
          cudaMalloc((void**)&dev_bools, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_bools failed!");
          cudaMalloc((void**)&dev_negbools, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_negbools failed!");
          cudaMalloc((void**)&dev_scanned, nearestPow2 * sizeof(int));
          checkCUDAError("cudaMalloc dev_scanned failed!");
          cudaMalloc((void**)&dev_writingIndices, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_writingIndices failed!");
          cudaMalloc((void**)&dev_indices, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_indices failed!");
          cudaMalloc((void**)&dev_odata, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_odata failed!");
          cudaMalloc((void**)&dev_scatterBools, n * sizeof(int));
          checkCUDAError("cudaMalloc dev_scatterBools failed!");

          cudaMemset(dev_scanned, 0, nearestPow2 * sizeof(int));
          checkCUDAError("cudaMemset dev_scanned failed!");

          cudaMemset(dev_scatterBools, 1, n * sizeof(int));
          checkCUDAError("cudaMemset dev_scatterBools failed!");

          cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
          checkCUDAError("cudaMemcpy to device failed!");

          timer().startGpuTimer();

          dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

          const int numBits = 32;
          for (int i = 0; i < numBits; i++) {
            // split based on the ith least significant bit
            // isolate current bit
            kernMapToBits<<<fullBlocksPerGrid, threadsPerBlock>>>(n, i, dev_idata, dev_bools);
            checkCUDAError("kernMapToBits failed!");

            // Negate the bit values as bools
            kernNegateArray<<<fullBlocksPerGrid, threadsPerBlock>>>(n, dev_bools, dev_negbools);
            checkCUDAError("kernNegateArray failed!");

            // Scan negated bools
            cudaMemcpy(dev_scanned, dev_negbools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy to device failed!");
            // upsweep
            for (int d = 0; d < iters; d++) {
              int currOffset = 1 << d;
              dim3 fullBlocksPerGridScan((nearestPow2 / (currOffset * 2) + blockSize - 1) / blockSize);
              kernUpSweep<<<fullBlocksPerGridScan, threadsPerBlock>>> (nearestPow2, currOffset, dev_scanned);
            }

            // Set last value after upsweep to 0
            cudaMemset(dev_scanned + nearestPow2 - 1, 0, sizeof(int));

            //downsweep
            for (int d = iters - 1; d >= 0; d--) {
              int currOffset = 1 << d;
              dim3 fullBlocksPerGridScan((nearestPow2 / (currOffset * 2) + blockSize - 1) / blockSize);
              kernDownSweep<<<fullBlocksPerGridScan, threadsPerBlock>>> (nearestPow2, currOffset, dev_scanned);
            }

            // Compute total number of falses
            int totalFalses;
            cudaMemcpy(&totalFalses, dev_scanned + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int lastNegBool;
            cudaMemcpy(&lastNegBool, dev_negbools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            totalFalses += lastNegBool;

            // Compute writing indices
            kernGetWritingIndices<<<fullBlocksPerGrid, threadsPerBlock>>>(n, dev_scanned, dev_writingIndices, totalFalses);
            checkCUDAError("kernGetWritingIndices failed!");

            // Get scatter indices
            kernGetScatterIndices<<<fullBlocksPerGrid, threadsPerBlock>>>(n, dev_bools, dev_writingIndices, dev_scanned, dev_indices);
            checkCUDAError("kernGetScatterIndices failed!");

            // Scatter
            Common::kernScatter <<<fullBlocksPerGrid, threadsPerBlock >>>(n, dev_odata, dev_idata, dev_scatterBools, dev_indices);
            checkCUDAError("kernScatter failed!");

            // Swap idata and odata
            int* temp = dev_idata;
            dev_idata = dev_odata;
            dev_odata = temp;
          }

          timer().endGpuTimer();

          // Copy back to host
          cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
          checkCUDAError("cudaMemcpy to host failed!");

          // Free all device arrays
          cudaFree(dev_idata);
          cudaFree(dev_bools);
          cudaFree(dev_negbools);
          cudaFree(dev_scanned);
          cudaFree(dev_writingIndices);
          cudaFree(dev_indices);
          cudaFree(dev_odata);
          cudaFree(dev_scatterBools);
        }
    }
}
