#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int sum = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = sum;
                sum += idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int writeOffset = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[writeOffset] = idata[i];
                    writeOffset++;
                }
            }
            // TODO
            timer().endCpuTimer();
            return writeOffset;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* temp = new int[n];
            // create temp array
            int* scanned = new int[n];
            for (int i = 0; i < n; i++) {
                temp[i] = idata[i] == 0 ? 0 : 1;
            }
            
            // scan temp array
            int sum = 0;
            for (int i = 0; i < n; i++) {
              scanned[i] = sum;
              sum += temp[i];
            }

            // scatter
            for (int i = 0; i < n; i++) {
                if (temp[i]) {
                    odata[scanned[i]] = idata[i];
                }
            }
            int count = scanned[n - 1] + temp[n - 1];
            timer().endCpuTimer();
            return count;
        }

        // CPU sort for testing GPU radix sort
        void sort(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            memcpy(odata, idata, n * sizeof(int));
            std::sort(odata, odata + n);
            timer().endCpuTimer();
        }
    }
}
