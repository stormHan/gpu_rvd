
#include <points\kd_tree.h>

namespace Gpu_Rvd{

	Kdtree::Kdtree(coords_index_t dimension) :
		NearestNeighborSearch(dimension),
		bbox_max_(dimension),
		bbox_min_(dimension),
		m0_(max_index_t()),
		m1_(max_index_t()),
		m2_(max_index_t()),
		m3_(max_index_t()),
		m4_(max_index_t()),
		m5_(max_index_t()),
		m6_(max_index_t()),
		m7_(max_index_t()),
		m8_(max_index_t())
	{}

	Kdtree::~Kdtree(){}

	void Kdtree::set_points(
		index_t nb_points, const double* points
		){
		set_points(nb_points, points, dimension());
	}
}