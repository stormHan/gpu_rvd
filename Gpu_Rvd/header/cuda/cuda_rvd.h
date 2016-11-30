/*
 * /brief compute the Restricted Voronoi Diagram in CPU.
 *	
 */

#ifndef CUDA_RVD_H
#define CUDA_RVD_H

#include <basic\common.h>
#include <cuda\cuda_common.h>
#include <cuda\cuda_stop_watcher.h>
#include <mesh\mesh.h>


namespace Gpu_Rvd{

	class CudaRestrictedVoronoiDiagram{
	public:
		CudaRestrictedVoronoiDiagram();

		/*
		 * \brief if the Mesh[m] and Points[p] store the nn in themselves, we can construct the 
		 *		  the RVD with Mesh and Points own.
		 */
		CudaRestrictedVoronoiDiagram(Mesh m, Points p);

		/*
		 * \brief Construts the RVD with Mesh, Points and NN information.
		 */
		CudaRestrictedVoronoiDiagram(Mesh m, Points p, index_t k, const index_t* points_nn, const index_t* facets_nn);

		/*
		 * \brief Destruction. now it does nothing.
		 */
		~CudaRestrictedVoronoiDiagram();

		/*
		 * \brief Calls the kernel function.
		 */
		void compute_Rvd();

	protected:
		enum DeviceMemoryMode{
			GLOBAL_MEMORY = 0,
			CONSTANT_MEMORY = 1,
			TEXTURE_MEMORY = 2
		};

		/*
		 * \brief Allocates the device memory and 
		 *	copies the data from host.
		 */
		void allocate_and_copy(DeviceMemoryMode mode);

		/*
		 * \brief Frees the dev_pointer.
		 */
		void free_memory();

		/*
		 * \brief Checks if some manipulation get error.
		 */
		void CheckCUDAError(const char *msg)
		{
			cudaError_t err = cudaGetLastError();
			if (cudaSuccess != err) {
				fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}

		}
	private:
		const double* vertex_;
		index_t vertex_nb_;

		const double* points_;
		index_t points_nb_;

		const index_t* facets_;
		index_t facet_nb_;

		index_t k_;
		const index_t* points_nn_;
		const index_t* facets_nn_;

		index_t dimension_;

		double* dev_vertex_;
		double* dev_points_;
		index_t* dev_facets_;
		index_t* dev_points_nn_;
		index_t* dev_facets_nn_;
		double* dev_ret;
		double* host_ret;
	};

}

#endif /* CUDA_RVD_H */