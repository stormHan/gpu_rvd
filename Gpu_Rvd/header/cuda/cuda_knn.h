/*
*	  GPU Knn Algorithm
*
*/

#ifndef H_CUDA_KNN
#define H_CUDA_KNN

#include <basic\common.h>
#include <basic\math_op.h>
#include <cuda\cuda_common.h>
#include <mesh\mesh.h>
#include "cuda.h"

namespace Gpu_Rvd{

	/*
	 * \brief Use k-dtree to find K nearest neighbors.
	 */
	class CudaKNearestNeighbor{

	public:
		CudaKNearestNeighbor(const Points& p, int k);

		CudaKNearestNeighbor(const Points& p, const Mesh& m, int k);

		~CudaKNearestNeighbor();

		void search(index_t* result);

		void set_reference(const Points& p);

		void set_query(const Points& p);

		void set_query(const Mesh& m);
	private:

		void knn(float* ref_host, int ref_width, float* query_host, int query_width,
			int height, int k, float* dist_host, int* ind_host);

		void memory_allc(int ref_nb, int query_nb){
			ref_ = (float*)malloc(ref_nb * dim_ * sizeof(float));
			query_ = (float*)malloc(query_nb * dim_ * sizeof(float));
			dist_ = (float*)malloc(query_nb * k_ * sizeof(float));
			ind_ = (int*)malloc(query_nb * k_ * sizeof(int));
		};

		float*	ref_;
		float*	query_;
		int		ref_nb_;
		int		query_nb_;
		int		dim_;
		int		k_;

		float*	dist_;
		int*	ind_;

	};
}

#endif /* H_CUDA_KNN */