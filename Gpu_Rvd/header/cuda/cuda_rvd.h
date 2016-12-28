/*
 * /brief compute the Restricted Voronoi Diagram in CPU.
 *	
 */

#ifndef CUDA_RVD_H
#define CUDA_RVD_H

#include <basic\common.h>
#include <cuda\cuda_common.h>
#include <cuda\cuda_math.h>
#include <cuda\cuda_stop_watcher.h>
#include <cuda\cuda_polygon.h>
#include <cuda\cuda_knn.h>
#include <mesh\mesh.h>
#include <mesh\mesh_io.h>
#include "cuda.h"
#include <fstream>
#include <iomanip>

namespace Gpu_Rvd{

	class CudaRestrictedVoronoiDiagram{
	public:

		/*
		 * \brief if the Mesh[m] and Points[p] store the nn in themselves, we can construct the 
		 *		  the RVD with Mesh and Points own.
		 */
		CudaRestrictedVoronoiDiagram(Mesh m, Points p, int iter, int k = 20);

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

		bool is_store(){ return is_store_; }
		void set_if_store(bool x) { is_store_ = x; }

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
		 * \brief Uses CudaKNearestNeighbor to find the knn of points and facets
		 */
		void knn_search();

		/*
		 * \brief Copies back the data from device to host.
		 */
		void copy_back();

		/*
		 * \breif Updatas the result from GPU to points for the next iteration
		 */
		void update_points();

		/*
		 * \brief Prints the return data for convenient debug.
		 */
		void print_return_data(const std::string filename) const;

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
		//CPU data
		const double* vertex_;
		index_t vertex_nb_;
		const double* points_;
		index_t points_nb_;
		const index_t* facets_;
		index_t facet_nb_;

		//Knn 
		index_t k_;
		index_t* points_nn_;
		index_t* facets_nn_;

		index_t dimension_;

		//GPU data
		double* dev_vertex_;
		double* dev_points_;
		index_t* dev_facets_;
		index_t* dev_points_nn_;
		index_t* dev_facets_nn_;
		double* dev_ret_;
		double* host_ret_;

		double* dev_seeds_info_;
		int*	dev_seeds_poly_nb;

		Mesh* mesh_;
		Points* x_;

		CudaKNearestNeighbor* knn_;
		int iter_nb_;

		bool is_store_;
		int store_filename_counter_;
	};

}

#endif /* CUDA_RVD_H */