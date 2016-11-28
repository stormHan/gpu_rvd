/*
	Nearest Neighbors implementation of mesh
*/

#ifndef MESH_NN_H
#define MESH_NN_H

#include <basic\common.h>
#include <basic\math_op.h>

#include <map>

namespace Gpu_Rvd{

	class Points_nn{
	public:
		/*
		 * \brief nearest neighbors manipulation utilities
		 */
		Points_nn();

		/*
		 *\brief destruction of the class
		 */
		~Points_nn(); 

		/*
		 * \brief initialize the k_ and malloc the memroy for index_
		 * \details must be called before all the operation, otherwise 
		 * you may get some errors
		 * \param[selfstore] if you want to store the nn informantion(indices and dists)
		 * you should let it true.
		 */
		void init(index_t k, coords_index_t dim, bool selfstore = false);
		
		/**
		 * /brief Sets the reference points.
		 */
		void set_points(index_t ref_nb, const double* points);

		/*
		 * \brief Mallocs the memroy of index_.
		 */
		bool malloc_index(index_t nb){
			if (selfstore_){
				index_ = (index_t*)malloc(sizeof(index_t) * nb * k_);
				return true;
			}
			else{
				fprintf(stderr, "cannot malloc memory for index_ in this mode");
				return false;
			}
		}

		/*
		* \brief Mallocs the memroy of dists_
		*/
		bool malloc_dist2(index_t nb){
			if (selfstore_){
				dist2_ = (double*)malloc(sizeof(double) * nb * k_);
				return true;
			}
			else{
				fprintf(stderr, "cannot malloc memory for dists_ in this mode");
				return false;
			}
		}

		/**
		 * \brief Gets the k nearest neighbors of the specific point.
		 * we will not self-store the infomation in this function.
		 */
		void get_nearest_neighbors(const double* query, index_t* neighbors, index_t neighbors_nb, double* dists);

		/*
		 * \brief Gets the k nearest neighbors in self-store mode.
		 */
		void get_nearest_neighbors(const double* query, index_t neighbors_nb, index_t offset);

		/*
		 * \brief Gets the required neighbor_nb
		 */
		const index_t get_k() const{
			return k_;
		}

		/*
		 * \brief Clears the data.
		 */
		void clear(){
			if (index_ != nil){
				free(index_);
				index_ = nil;
			}
			if (dist2_ != nil){
				free(dist2_);
				dist2_ = nil;
			}
		}

		/*
		 * \brief Gets the stored index
		 */
		const index_t* get_index() const{
			if (index_ == nil){
				fprintf(stderr, "the indices is nil");
			}
			return index_;
		}

	private:
		index_t k_;
		index_t* index_;
		double* dist2_;

		index_t ref_nb_;
		const double* ref_;
		coords_index_t stride_;

		bool selfstore_;
	};
	
}

#endif /* MESH_NN_H */