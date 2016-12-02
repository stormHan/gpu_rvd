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

	//Set cuda global variables.
	//Adds the RVD information to seeds_information to iterate 
	//the points' position.
	__device__
		double* g_seeds_information;

	//Records how many polygons belonged to each point.
	__device__
		int*	g_seeds_polygon_nb;
	
}

#endif /* CUDA_COMMON_H */