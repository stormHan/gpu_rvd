/*
 * \header Cuda Common.
 */

#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <basic\common.h>
#include <cuda_runtime.h>

namespace Gpu_Rvd{

#define DOUBLE_SIZE sizeof(double)
#define FLOAT_SIZE sizeof(double)
#define INT_SIZE sizeof(int)

	const index_t CUDA_Stack_size = 10;
}

#endif /* CUDA_COMMON_H */