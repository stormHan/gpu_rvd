
#include <points\nn_search.h>
#include <basic\numeric.h>
#include <points\kd_tree.h>

namespace Gpu_Rvd{

	NearestNeighborSearch::NearestNeighborSearch(coords_index_t dimension) :
		dimension_(dimension),
		nb_points_(0),
		stride_(0),
		points_(nil),
		exact_(0)
	{
	}

	void NearestNeighborSearch::get_nearest_neighbors(
		index_t nb_neighbors,
		index_t query_point,
		index_t* neighbors,
		double* neighbors_sq_dist
		) const {
		get_nearest_neighbors(
			nb_neighbors,
			point_ptr(query_point),
			neighbors,
			neighbors_sq_dist
			);
	}

	void NearestNeighborSearch::set_points(
		index_t nb_points, const double* points
		) {
		nb_points_ = nb_points;
		points_ = points;
		stride_ = dimension_;
	}

	NearestNeighborSearch::~NearestNeighborSearch() {
	}

	NearestNeighborSearch* NearestNeighborSearch::create(
		coords_index_t dimension, const std::string& name_in
		){

		return new Kdtree(dimension);
	}

	bool NearestNeighborSearch::stride_supported() const{
		return false;
	}
}