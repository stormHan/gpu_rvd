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

#define MAX_PITCH_VALUE_IN_BYTES       262144
#define MAX_TEXTURE_WIDTH_IN_BYTES     65536
#define MAX_TEXTURE_HEIGHT_IN_BYTES    32768
#define MAX_PART_OF_FREE_MEMORY_USED   0.9
#define BLOCK_DIM					   16

	const index_t CUDA_Stack_size = 10;

	//Set cuda global variables.
	//Adds the RVD information to seeds_information to iterate 
	//the points' position.
	__device__
		double* g_seeds_information;

	//Records how many polygons belonged to each point.
	__device__
		int*	g_seeds_polygon_nb;
	
	__constant__
		double	c_vertex[500];
	
	__constant__
		double c_points[500];

	__constant__
		index_t c_points_nn[3240];

	
}

#endif /* CUDA_COMMON_H */