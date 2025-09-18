#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {

        // Block variables
        #define blockSize 128
        dim3 threadsPerBlock(blockSize);

        // Data buffers
        int* dev_idata;
        int* dev_bools;
        int* dev_scanned;
        int* dev_indices;
        int* dev_odata;
        int* dev_blockSums;

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int iters = ilog2ceil(n);
            int nearestPow2 = 1 << iters;

            // set up device arrays to the nearest power of 2
            cudaMalloc((void**)&dev_indices, nearestPow2 * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMemset(dev_indices, 0, nearestPow2 * sizeof(int));
            checkCUDAError("cudaMemset dev_indices failed!");
            cudaMemcpy(dev_indices, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed!");

            timer().startGpuTimer();

            // upsweep
            for (int d = 0; d < iters; d++) {
              // calculate power of 2 offset with bitshift
              int currOffset = 1 << d;
              // Only call the number of threads that actually need to write no values in the current sweep level
              dim3 fullBlocksPerGrid((nearestPow2 / (currOffset * 2) + blockSize - 1) / blockSize);
              kernUpSweep<<<fullBlocksPerGrid, threadsPerBlock>>> (nearestPow2, currOffset, dev_indices);
            }

            // Set last value after upsweep to 0
            cudaMemset(dev_indices + nearestPow2 - 1, 0, sizeof(int));
            // downsweep
            for (int d = iters - 1; d >= 0; d--) {
              // calculate power of 2 offset with bitshift
              int currOffset = 1 << d;
              // Only call the number of threads that actually need to write no values in the current sweep level
              dim3 fullBlocksPerGrid((nearestPow2 / (currOffset * 2) + blockSize - 1) / blockSize);
              kernDownSweep<<<fullBlocksPerGrid, threadsPerBlock >>> (nearestPow2, currOffset, dev_indices);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_indices, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to host failed!");

            // Free device arrays
            cudaFree(dev_indices);
            checkCUDAError("cudaFree dev_indices failed!");

        }

        __global__ void kernSharedMemScan(int n, int* odata, int* idata, int* blockSums) {
          extern __shared__ int temp[];
          // Only one block to maintain shared memory
          int index = threadIdx.x;
          int blockStartIndex = blockIdx.x * 2048;
          int offset = 1;
          //load entire input into shared mem
          temp[2 * index] = idata[2 * index + blockStartIndex];
          temp[2 * index + 1] = idata[2 * index + blockStartIndex + 1];
          // upsweep
          for (int d = n >> 1; d > 0; d >>= 1) {
            __syncthreads();
            if (index < d) {
              int leftChild = offset * (2 * index + 1) - 1;
              int parent = offset * (2 * index + 2) - 1;
              temp[parent] += temp[leftChild];
            }
            offset *= 2;
          }
          __syncthreads();
          // capture last elem in block sums then zero it out zero out last element of temp array
          if (index == 0) {
            blockSums[blockIdx.x] = temp[n - 1];
            temp[n - 1] = 0;
          }
          //downsweep
          for (int d = 1; d < n; d *= 2) {
            offset >>= 1;
            __syncthreads();
            if (index < d) {
              int leftChild = offset * (2 * index + 1) - 1;
              int parent = offset * (2 * index + 2) - 1;
              int saved = temp[leftChild];
              temp[leftChild] = temp[parent];
              temp[parent] += saved;
            }
          }
          __syncthreads();
          odata[2 * index + blockStartIndex] = temp[2 * index];
          odata[2 * index + blockStartIndex + 1] = temp[2 * index + 1];
          
        }

        __global__ void kernAddBlockSums(int n, int* odata, int* blockSums) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n) {
            return;
          }
          int blockSumIndex = index / 2048;
          odata[index] += blockSums[blockSumIndex];
        }

        void sharedMemScan(int n, int* odata, const int* idata) {
          // max allowed in shared memory of one block
          if (n > 1 << 22) {
            return;
          }
          int iters = ilog2ceil(n);
          int nearestPow2 = 1 << iters;

          // each individual block can handle 2048 data points
          const int SPLIT = 2048;

          // set up device arrays to the nearest power of 2
          cudaMalloc((void**)&dev_indices, nearestPow2 * sizeof(int));
          checkCUDAError("cudaMalloc dev_indices failed!");
          cudaMalloc((void**)&dev_odata, nearestPow2 * sizeof(int));
          checkCUDAError("cudaMalloc dev_odata failed!");
          cudaMalloc((void**)&dev_blockSums, (((nearestPow2 + SPLIT - 1) / SPLIT)) * sizeof(int));
          checkCUDAError("cudaMalloc dev_blockSums failed!");
          cudaMalloc((void**)&dev_scanned, (((nearestPow2 + SPLIT - 1) / SPLIT)) * sizeof(int));
          checkCUDAError("cudaMalloc dev_scanned failed!");

          cudaMemset(dev_indices, 0, nearestPow2 * sizeof(int));
          checkCUDAError("cudaMemset dev_indices failed!");
          cudaMemcpy(dev_indices, idata, n * sizeof(int), cudaMemcpyHostToDevice);
          checkCUDAError("cudaMemcpy to device failed!");

          timer().startGpuTimer();
          
          const int maxThreadsPerBlock = 1024;
          int scannedSoFar = 0;
          int blocksNeeded = (nearestPow2 + 2048 - 1) / 2048;

          if (blocksNeeded == 1) {
            // only need one block
            kernSharedMemScan <<<1, nearestPow2 / 2, nearestPow2 * sizeof(int) >> > (nearestPow2, dev_odata, dev_indices, dev_blockSums);
          }
          else {
            // need multiple blocks and to scan block sums
            kernSharedMemScan <<<blocksNeeded, 1024, 1024 * 2 * sizeof(int) >> > (2048, dev_odata, dev_indices, dev_blockSums);
            kernSharedMemScan<<< 1, (blocksNeeded + 1) / 2, blocksNeeded * sizeof(int) >>> (blocksNeeded, dev_scanned, dev_blockSums, dev_blockSums);
            
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            kernAddBlockSums <<<fullBlocksPerGrid , threadsPerBlock >>> (n, dev_odata, dev_scanned);
          }

          timer().endGpuTimer();

          // copy data back over
          cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
          checkCUDAError("cudaMemcpy to host failed!");

          // free
          cudaFree(dev_indices);
          cudaFree(dev_odata);
          cudaFree(dev_blockSums);
        }



        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // TODO

            int iters = ilog2ceil(n);
            int nearestPow2 = 1 << iters;
            // set up device arrays to the nearest power of 2
            cudaMalloc((void**)&dev_bools, nearestPow2 * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_idata, nearestPow2 * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_indices, nearestPow2 * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMalloc((void**)&dev_odata, nearestPow2 * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMemset(dev_indices, 0, nearestPow2 * sizeof(int));
            checkCUDAError("cudaMemset dev_indices failed!");
            cudaMemset(dev_bools, 0, nearestPow2 * sizeof(int));
            checkCUDAError("cudaMemset dev_bools failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed!");

            timer().startGpuTimer();

            // map
            dim3 fullBlocksPerGrid((nearestPow2 + blockSize - 1) / blockSize);
            Common::kernMapToBoolean<<<(n + blockSize - 1) / blockSize, threadsPerBlock>>> (n, dev_bools, dev_idata);

            // scan
            
            // Copy the bools to the indices array for scan
            cudaMemcpy(dev_indices, dev_bools, nearestPow2 * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy dev_bools to dev_indices failed!");

            // upsweep
            for (int d = 0; d < iters; d++) {
              // calculate power of 2 offset with bitshift
              int currOffset = 1 << d;
              // Only call the number of threads that actually need to write no values in the current sweep level
              dim3 fullBlocksPerGrid((nearestPow2 / (currOffset * 2) + blockSize - 1) / blockSize);
              kernUpSweep<<<fullBlocksPerGrid, threadsPerBlock>>> (nearestPow2, currOffset, dev_indices);
            }

            // Set last value after upsweep to 0
            cudaMemset(dev_indices + nearestPow2 - 1, 0, sizeof(int));

            //downsweep
            for (int d = iters - 1; d >= 0; d--) {
              // calculate power of 2 offset with bitshift
              int currOffset = 1 << d;
              // Only call the number of threads that actually need to write no values in the current sweep level
              dim3 fullBlocksPerGrid((nearestPow2 / (currOffset * 2) + blockSize - 1) / blockSize);
              kernDownSweep<<<fullBlocksPerGrid, threadsPerBlock>>> (nearestPow2, currOffset, dev_indices);
            }

            // scatter
            Common::kernScatter<<<(n + blockSize - 1) / blockSize, threadsPerBlock >>> (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

            // figure out num elements
            int lastIndex;
            int lastBool;
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int size = lastIndex + lastBool;
            cudaMemcpy(odata, dev_odata, size * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to host failed!");

            // Free device arrays
            cudaFree(dev_idata);
            checkCUDAError("cudaFree dev_idata failed!");
            cudaFree(dev_bools);
            checkCUDAError("cudaFree dev_bools failed!");
            cudaFree(dev_indices);
            checkCUDAError("cudaFree dev_indices failed!");
            cudaFree(dev_odata);
            checkCUDAError("cudaFree dev_odata failed!");

            return size;
        }
    }
}
