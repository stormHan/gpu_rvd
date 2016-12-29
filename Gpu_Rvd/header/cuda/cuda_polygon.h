/*
 *
 */

#ifndef CUDA_POLYGON_H
#define CUDA_POLYGON_H

namespace Gpu_Rvd{

	/*
	* \brief Cuda polygon vertex.
	* \detials x, y, z is the spatial position of a vertxt.
	*	w is the weight of a vextex. neigh_s is the nearby seed
	* of the current point.
	*/
	struct CudaVertex
	{
		double x;
		double y;
		double z;
		double w;

		int neigh_s = -1;
	};

	/*
	* \brief Cuda polygon. A smart data stuction to store the clipped triangle.
	*/
	struct CudaPolygon
	{
		CudaVertex vertex[20];
		index_t vertex_nb;
	};
}

#endif /* CUDA_POLYGON_H */