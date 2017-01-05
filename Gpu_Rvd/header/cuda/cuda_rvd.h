/*
 * /brief compute the Restricted Voronoi Diagram in CPU.
 *	
 */

#ifndef CUDA_RVD_H
#define CUDA_RVD_H

#include <basic\common.h>
#include <basic\process.h>
#include <cuda\cuda_common.h>
#include <cuda\cuda_math.h>
#include <cuda\cuda_stop_watcher.h>
#include <cuda\cuda_polygon.h>
#include <cuda\cuda_knn.h>
#include <mesh\mesh.h>
#include <mesh\mesh_io.h>
#include <points\nn_search.h>
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
		CudaRestrictedVoronoiDiagram(Mesh* m, Points* p, int iter, int k = 20, int fk = 4);

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

		/**
		 * \brief Updates the points' neighbors inside.
		 */
		void update_neighbors();

		/**
		 * \brief multi-thread friendly function to search and store
		 * the neighbors.
		 */
		void store_neighbors_CB(index_t v);

		/**
		 * \brief multi-thread friendly function to search for facets
		 */
		void store_f_neighrbors_CB(index_t v);

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

		template <typename T>
		bool result_print(std::string filename, const T* address, int total_nb, int line_nb){
			std::ofstream out(filename.c_str());
			if (!out){
				std::cerr
					<< "Could not create file <"
					<< filename
					<< ">"
					<< std::endl;
				return false;
			}
			out << "# filename: " << filename << std::endl;
			int c = 0;
			for (int i = 0; i < total_nb; ++i){
				out << address[i] << " ";
				c++;
				if (line_nb == c){
					out << std::endl;
					c = 0;
				}
				
			}
			return true;
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

		//CudaKNearestNeighbor* knn_;
		NearestNeighborSearch_var NN_;

		int iter_nb_;

		bool is_store_;
		int store_filename_counter_;

		index_t fk_;
		double* facets_center_;
	};

}

#endif /* CUDA_RVD_H */