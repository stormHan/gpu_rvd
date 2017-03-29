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
#include <stack>
#include <unordered_set>

namespace Gpu_Rvd{

	class CudaRestrictedVoronoiDiagram{
	public:

		/*
		 * \brief if the Mesh[m] and Points[p] store the nn in themselves, we can construct the 
		 *		  the RVD with Mesh and Points own.
		 */
		CudaRestrictedVoronoiDiagram(Mesh* m, Points* p, int iter, std::vector<int> sample_facet, int k = 20, int f_k = 1);

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

		/**
		 * \brief checks if all the facet is finished.
		 */
		bool check_task_finished(std::vector<std::stack<int>> to_visited){
			for (index_t t = 0; t < facet_nb_; ++t){
				if (to_visited[t].size() > 0) 
					return false;
			}
			return true;
		}
		
		/**
		 * \brief insert the to_visited stack for the next Kernel.
		 */
		void insert_to_visited(int* retidx, index_t data_size){
			for (index_t t = 0; t < facet_nb_; ++t){
				for (index_t i = 0; i < 5; ++i){
					for (index_t ii = 0; ii < data_size; ++ii){
						int cur = retidx[(t * 5 + i) * data_size + ii];
						if (cur < -1) break;
						if (cur == -1) continue;
						if (has_visited[t].find(cur) == has_visited[t].end()){
							has_visited[t].insert(cur);
							to_visited[t].push(cur);
						}
					}
				}
			}
		}

		/**
		 *\brief Puts the data into *next from to_visited stack
		 */
		void go_next(int* next, index_t data_size){
			for (int i = 0; i < facet_nb_; ++i){
				index_t c = 0;
				while (!to_visited[i].empty() && c < data_size){
					next[i * data_size + 1 + c] = to_visited[i].top();
					to_visited[i].pop();
					c++;
				}
				next[i * data_size] = c;
				c = 0;
			}
		}

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
		index_t f_k_;
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
		int* dev_retidx;
		int* host_retidx;

		double* dev_seeds_info_;
		int*	dev_seeds_poly_nb;

		Mesh* mesh_;
		Points* x_;

		//CudaKNearestNeighbor* knn_;
		NearestNeighborSearch_var NN_;

		int iter_nb_;

		bool is_store_;
		int store_filename_counter_;

		std::vector<int> sample_facet_;
		std::vector<std::stack<int>> to_visited;
		std::vector<std::unordered_set<int>> has_visited;

	};

}

#endif /* CUDA_RVD_H */