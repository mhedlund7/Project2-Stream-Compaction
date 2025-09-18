#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Radix {
        StreamCompaction::Common::PerformanceTimer& timer();

        void radix(int n, int *odata, const int *idata);

        void setBlockSize(int newBlockSize);
    }
}
