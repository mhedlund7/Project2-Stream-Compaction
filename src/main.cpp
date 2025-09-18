/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include "testing_helpers.hpp"
#include <iostream>
#include <fstream>

const int SIZE = 1 << 12; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];


// Functions for data collection
static void setAllBlockSizes(int blockSize) {
  StreamCompaction::Naive::setBlockSize(blockSize);
  StreamCompaction::Efficient::setBlockSize(blockSize);
  StreamCompaction::Radix::setBlockSize(blockSize);
}

static void fillScanArray(int n, int* a) {
  genArray(n - 1, a, 50);
  a[n - 1] = 0;
}

static void fillCompactArray(int n, int* a) {
  genArray(n - 1, a, 4);
  a[n - 1] = 0;
}

static void fillRadixArray(int n, int* a) {
  genArray(n - 1, a, 2048);
}

static double getAvgCPUScanData(int n, int* o, int* in) {
  const int numRuns = 10;
  double sum = 0.0;
  for (int i = 0; i < numRuns; i++) {
    StreamCompaction::CPU::scan(n, o, in);
    sum += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
  }
  return sum / numRuns;
}

static double getAvgThrustScanData(int n, int* o, int* in) {
  const int numRuns = 10;
  double sum = 0.0;
  for (int i = 0; i < numRuns; i++) {
    StreamCompaction::Thrust::scan(n, o, in);
    sum += StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
  }
  return sum / numRuns;
}

static double getAvgNaiveScanData(int n, int* o, int* in) {
  const int numRuns = 10;
  double sum = 0.0;
  for (int i = 0; i < numRuns; i++) {
    StreamCompaction::Naive::scan(n, o, in);
    sum += StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
  }
  return sum / numRuns;
}

static double getAvgEfficientScanData(int n, int* o, int* in) {
  const int numRuns = 10;
  double sum = 0.0;
  for (int i = 0; i < numRuns; i++) {
    StreamCompaction::Efficient::scan(n, o, in);
    sum += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
  }
  return sum / numRuns;
}

static double getAvgEfficientSharedMemScanData(int n, int* o, int* in) {
  const int numRuns = 10;
  double sum = 0.0;
  for (int i = 0; i < numRuns; i++) {
    StreamCompaction::Efficient::sharedMemScan(n, o, in);
    sum += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
  }
  return sum / numRuns;
}

static double getAvgCPUCompactWithoutScanData(int n, int* o, int* in) {
  const int numRuns = 10;
  double sum = 0.0;
  for (int i = 0; i < numRuns; i++) {
    StreamCompaction::CPU::compactWithoutScan(n, o, in);
    sum += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

  }
  return sum / numRuns;
}

static double getAvgCPUCompactWithScanData(int n, int* o, int* in) {
  const int numRuns = 10;
  double sum = 0.0;
  for (int i = 0; i < numRuns; i++) {
    StreamCompaction::CPU::compactWithScan(n, o, in);
    sum += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
  }
  return sum / numRuns;
}

static double getAvgEfficientCompactData(int n, int* o, int* in) {
  const int numRuns = 10;
  double sum = 0.0;
  for (int i = 0; i < numRuns; i++) {
    StreamCompaction::Efficient::compact(n, o, in);
    sum += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
  }
  return sum / numRuns;
}

static double getAvgRadixnData(int n, int* o, int* in) {
  const int numRuns = 10;
  double sum = 0.0;
  for (int i = 0; i < numRuns; i++) {
    StreamCompaction::Radix::radix(n, o, in);
    sum += StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation();
  }
  return sum / numRuns;
}

// Actually collect the data
void collectData() {
  const char* dataPath = "data.csv";
  std::ofstream dataFile(dataPath);
  if (!dataFile) {
    return;
  }
  dataFile << "implementation,block_size,array_size,power_of_two,time_ms\n";

  const int MAXN = (1 << 27) +1;
  int* a = new int[MAXN];
  int* b = new int[MAXN];
  int* c = new int[MAXN];

  // Block Size vs Time data using N = 20

  const int blockSizes[] = { 64, 128, 256, 512, 1024 };
  int powerOfTwoFlag = 1;
  int currSize = 1 << 20;

  for (int blockSize : blockSizes) {
    setAllBlockSizes(blockSize);

    // CPUScan
    fillScanArray(currSize, a);
    zeroArray(currSize, c);
    double t = getAvgCPUScanData(currSize, c, a);
    dataFile << "CPUScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    // NaiveScan
    zeroArray(currSize, c);
    t = getAvgNaiveScanData(currSize, c, a);
    dataFile << "NaiveScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //EfficientScan
    zeroArray(currSize, c);
    t = getAvgEfficientScanData(currSize, c, a);
    dataFile << "EfficientScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //EfficientStreamCompaction
    fillCompactArray(currSize, a);
    zeroArray(currSize, c);
    t = getAvgEfficientCompactData(currSize, c, a);
    dataFile << "EfficientCompact," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //CPUCompactWithScan
    zeroArray(currSize, c);
    t = getAvgCPUCompactWithScanData(currSize, c, a);
    dataFile << "CPUCompactWithScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //CPUCompactWithoutuScan
    zeroArray(currSize, c);
    t = getAvgCPUCompactWithoutScanData(currSize, c, a);
    dataFile << "CPUCompactWithoutuScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //Radix
    fillRadixArray(currSize, a);
    zeroArray(currSize, c);
    t = getAvgNaiveScanData(currSize, c, a);
    dataFile << "RadixSort," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";
  }

  // Array Size vs Time

  int blockSize = 256;
  setAllBlockSizes(blockSize);
  const int arraySizes[] = { 1 << 8, 1 << 10, 1 << 13, 1 << 15, 1 << 17, 1 << 20, 1 << 22, 1 << 24, 1 << 26};
  const int numSizes = 9;

  // Powers of 2
  for (int i = 0; i < numSizes; i++) {

    int currSize = arraySizes[i];

    // CPUScan
    fillScanArray(currSize, a);
    zeroArray(currSize, c);
    double t = getAvgCPUScanData(currSize, c, a);
    dataFile << "CPUScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    // NaiveScan
    zeroArray(currSize, c);
    t = getAvgNaiveScanData(currSize, c, a);
    dataFile << "NaiveScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    // ThrustScan
    zeroArray(currSize, c);
    t = getAvgThrustScanData(currSize, c, a);
    dataFile << "ThrustScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //EfficientScan
    zeroArray(currSize, c);
    t = getAvgEfficientScanData(currSize, c, a);
    dataFile << "EfficientScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //EfficientSharedMemScan
    zeroArray(currSize, c);
    StreamCompaction::Efficient::setMemoryBankOptimized(0);
    t = getAvgEfficientSharedMemScanData(currSize, c, a);
    dataFile << "EfficientSharedMemScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //EfficientBankOptimizedSharedMemScan
    zeroArray(currSize, c);
    StreamCompaction::Efficient::setMemoryBankOptimized(1);
    t = getAvgEfficientSharedMemScanData(currSize, c, a);
    dataFile << "EfficientBankOptimizedSharedMemScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //EfficientStreamCompaction
    fillCompactArray(currSize, a);
    zeroArray(currSize, c);
    t = getAvgEfficientCompactData(currSize, c, a);
    dataFile << "EfficientCompact," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //CPUCompactWithScan
    zeroArray(currSize, c);
    t = getAvgCPUCompactWithScanData(currSize, c, a);
    dataFile << "CPUCompactWithScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //CPUCompactWithoutuScan
    zeroArray(currSize, c);
    t = getAvgCPUCompactWithoutScanData(currSize, c, a);
    dataFile << "CPUCompactWithoutuScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //Radix
    fillRadixArray(currSize, a);
    zeroArray(currSize, c);
    t = getAvgNaiveScanData(currSize, c, a);
    dataFile << "RadixSort," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";
  }

  // Non Powers of 2
  powerOfTwoFlag = 0;
  for (int i = 0; i < numSizes; i++) {

    int currSize = arraySizes[i] - 3;

    // CPUScan
    fillScanArray(currSize, a);
    zeroArray(currSize, c);
    double t = getAvgCPUScanData(currSize, c, a);
    dataFile << "CPUScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    // NaiveScan
    zeroArray(currSize, c);
    t = getAvgNaiveScanData(currSize, c, a);
    dataFile << "NaiveScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    // ThrustScan
    zeroArray(currSize, c);
    t = getAvgThrustScanData(currSize, c, a);
    dataFile << "ThrustScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //EfficientScan
    zeroArray(currSize, c);
    t = getAvgEfficientScanData(currSize, c, a);
    dataFile << "EfficientScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //EfficientSharedMemScan
    zeroArray(currSize, c);
    StreamCompaction::Efficient::setMemoryBankOptimized(0);
    t = getAvgEfficientSharedMemScanData(currSize, c, a);
    dataFile << "EfficientSharedMemScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //EfficientBankOptimizedSharedMemScan
    zeroArray(currSize, c);
    StreamCompaction::Efficient::setMemoryBankOptimized(1);
    t = getAvgEfficientSharedMemScanData(currSize, c, a);
    dataFile << "EfficientBankOptimizedSharedMemScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //EfficientStreamCompaction
    fillCompactArray(currSize, a);
    zeroArray(currSize, c);
    t = getAvgEfficientCompactData(currSize, c, a);
    dataFile << "EfficientCompact," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //CPUCompactWithScan
    zeroArray(currSize, c);
    t = getAvgCPUCompactWithScanData(currSize, c, a);
    dataFile << "CPUCompactWithScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //CPUCompactWithoutuScan
    zeroArray(currSize, c);
    t = getAvgCPUCompactWithoutScanData(currSize, c, a);
    dataFile << "CPUCompactWithoutuScan," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";

    //Radix
    fillRadixArray(currSize, a);
    zeroArray(currSize, c);
    t = getAvgNaiveScanData(currSize, c, a);
    dataFile << "RadixSort," << blockSize << "," << currSize << "," << powerOfTwoFlag << "," << t << "\n";
  }

  dataFile.close();
}

int main(int argc, char* argv[]) {

    // CollectData
    //collectData();

    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    StreamCompaction::Efficient::setMemoryBankOptimized(1);

    zeroArray(SIZE, c);
    printDesc("shared mem work-efficient scan, power-of-two");
    StreamCompaction::Efficient::sharedMemScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("shared mem work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::sharedMemScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("******** RADIX TESTS ********\n");
    printf("*****************************\n");

    genArray(SIZE - 1, a, 200);

    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu sort, power-of-two");
    StreamCompaction::CPU::sort(SIZE, b, a);
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("radix, power-of-two");
    StreamCompaction::Radix::radix(SIZE, c, a);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, b);
    printDesc("cpu sort, non-power-of-two");
    StreamCompaction::CPU::sort(NPOT, b, a);
    printArray(NPOT, b, true);

    zeroArray(SIZE, c);
    printDesc("radix, non-power-of-two");
    StreamCompaction::Radix::radix(NPOT, c, a);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}

