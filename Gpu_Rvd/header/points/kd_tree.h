
#ifndef H_POINTS_KDTREE
#define H_POINTS_KDTREE

#include <basic\common.h>
#include <points\nn_search.h>
#include <algorithm>

/*
 * \file header/points/kdtree.h
 * \brief An implementation of NearestNeighborSearch with a kd-tree
 */

namespace Gpu_Rvd{

	class Kdtree : public NearestNeighborSearch{
	public:
		Kdtree(coords_index_t dimension);

		virtual void set_points(index_t nb_points, const double* points);

		void get_nearest_neighbors(
			index_t nb_neighbors,
			const double* query_point,
			index_t* neighbors,
			double* neighbors_sq_dist
			) const {}

		virtual void set_points(
			index_t nb_points, const double* points, index_t stride
			);

	protected:
		virtual ~Kdtree();

		/*
		 * \brief Number of points stored in the leafs of the tree.
		 */
		static const index_t MAX_LEAF_SIZE = 16;


	protected:
		std::vector<index_t> point_index_;
		std::vector<coords_index_t> splitting_coord_;
		std::vector<double> splitting_val_;
		std::vector<double> bbox_min_;
		std::vector<double> bbox_max_;

		index_t m0_, m1_, m2_, m3_, m4_, m5_, m6_, m7_, m8_;
	};

}
#endif /* H_POINTS_KDTREE */