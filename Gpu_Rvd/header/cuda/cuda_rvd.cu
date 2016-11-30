/*
 * /brief Implementation of computing Restricted Voronoi Diagram.
 */

#include <cuda\cuda_rvd.h>

namespace Gpu_Rvd{

	CudaRestrictedVoronoiDiagram::CudaRestrictedVoronoiDiagram(Mesh m, Points p) : 
		vertex_(m.v_ptr()),
		vertex_nb_(m.get_vertex_nb()),
		points_(p.v_ptr()),
		points_nb_(p.get_vertex_nb()),
		facets_(m.f_ptr()),
		facet_nb_(m.get_facet_nb()),
		k_(p.get_k()),
		points_nn_(p.get_indices()),
		facets_nn_(m.get_indices()),
		dimension_(p.dimension()),
		dev_vertex_(nil),
		dev_points_(nil),
		dev_facets_(nil),
		dev_points_nn_(nil),
		dev_facets_nn_(nil),
		dev_ret(nil),
		host_ret(nil)
	{
	}

	CudaRestrictedVoronoiDiagram::CudaRestrictedVoronoiDiagram(Mesh m, Points p, index_t k, const index_t* points_nn, const index_t* facets_nn) :
		vertex_(m.v_ptr()),
		vertex_nb_(m.get_vertex_nb()),
		points_(p.v_ptr()),
		points_nb_(p.get_vertex_nb()),
		facets_(m.f_ptr()),
		facet_nb_(m.get_facet_nb()),
		k_(k),
		points_nn_(points_nn),
		facets_nn_(facets_nn),
		dimension_(p.dimension()),
		dev_vertex_(nil),
		dev_points_(nil),
		dev_facets_(nil),
		dev_points_nn_(nil),
		dev_facets_nn_(nil),
		dev_ret(nil),
		host_ret(nil)
	{
	}

	CudaRestrictedVoronoiDiagram::~CudaRestrictedVoronoiDiagram()
	{
	}

	__global__
		void kernel(
		double*			vertex,		index_t			vertex_nb,
		double*			points,		index_t			points_nb,
		index_t*		facets,		index_t			facets_nb,
		index_t*		points_nn,	index_t			k_p,
		index_t*		facets_nn,	index_t			k_f,
		double*			retdata
		){

	}

	__host__
	void CudaRestrictedVoronoiDiagram::compute_Rvd(){
		CudaStopWatcher watcher("compute_rvd");
		watcher.start();

		allocate_and_copy(GLOBAL_MEMORY);
		int threads = 512;
		int blocks = facet_nb_ / threads + ((facet_nb_ % threads) ? 1 : 0);
		kernel << < threads, blocks >> > (
			dev_vertex_, vertex_nb_,
			dev_points_, points_nb_,
			dev_facets_, facet_nb_,
			dev_points_nn_, k_,
			dev_facets_nn_, 1,
			dev_ret
			);
		CheckCUDAError("kernel function");
		watcher.synchronize();
		free_memory();
		watcher.print_elaspsed_time(std::cout);
	}

	__host__
	void CudaRestrictedVoronoiDiagram::allocate_and_copy(DeviceMemoryMode mode){
		switch (mode)
		{
		case GLOBAL_MEMORY:
		{
			//Allocate
			//Input data.
			cudaMalloc((void**)&dev_vertex_, DOUBLE_SIZE * vertex_nb_ * dimension_);
			cudaMalloc((void**)&dev_points_, DOUBLE_SIZE * points_nb_ * dimension_);
			cudaMalloc((void**)&dev_facets_, sizeof(index_t) * facet_nb_ * dimension_);
			cudaMalloc((void**)&dev_points_nn_, sizeof(index_t) * points_nb_ * k_);
			cudaMalloc((void**)&dev_facets_nn_, sizeof(index_t) * facet_nb_ * 1);

			//Output result.
			cudaMalloc((void**)&dev_ret, sizeof(double) * points_nb_ * 4);
			CheckCUDAError("Allocating device memory");

			//Copy
			cudaMemcpy(dev_vertex_, vertex_, DOUBLE_SIZE * vertex_nb_ * dimension_, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_points_, points_, DOUBLE_SIZE * points_nb_ * dimension_, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_facets_, facets_, sizeof(index_t) * facet_nb_ * dimension_, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_points_nn_, points_nn_, sizeof(index_t) * points_nb_ * k_, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_facets_nn_, facets_nn_, sizeof(index_t) * facet_nb_ * 1, cudaMemcpyHostToDevice);

			CheckCUDAError("Copying data from host to device");
		}
			break;
		case CONSTANT_MEMORY:
			break;
		case TEXTURE_MEMORY:
			break;
		default:
			break;
		}
	}

	__host__
	void CudaRestrictedVoronoiDiagram::free_memory(){
		cudaFree(dev_vertex_);
		cudaFree(dev_points_);
		cudaFree(dev_facets_);
		cudaFree(dev_points_nn_);
		cudaFree(dev_facets_nn_);
		cudaFree(dev_ret);

		if (host_ret != nil){
			free(host_ret);
			host_ret = nil;
		}
	}
}