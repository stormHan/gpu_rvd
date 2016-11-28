/*
	implementation of mesh_nn
*/

#include <mesh\mesh_nn.h>

namespace Gpu_Rvd{

	Points_nn::Points_nn() :
		k_(0),
		index_(nil),
		dist2_(nil),
		ref_nb_(0),
		ref_(nil),
		stride_(0),
		selfstore_(false)
	{};

	Points_nn::~Points_nn(){
		if (index_ != nil){
			free(index_);
			index_ = nil;
		}
		if (dist2_ != nil){
			free(dist2_);
			dist2_ = nil;
		}
	}

	void Points_nn::init(index_t k, coords_index_t dim, bool selfstore){
		k_ = k;
		stride_ = dim;
		selfstore_ = selfstore;
	}

	void Points_nn::set_points(index_t ref_nb, const double* points){
		ref_nb_ = ref_nb;
		ref_ = points;
	}

	void Points_nn::get_nearest_neighbors(const double* query, index_t* neighbors, index_t neighbors_nb, double* dists){
		if (ref_ == nil){
			fprintf(stderr, "the reference hasn't be initialized");
			return;
		}

		if (ref_nb_ < neighbors_nb){
			fprintf(stderr, "you've asked too much neighbors.");
		}
		std::map<double, index_t> dist_index_map;

		double d;
		for (index_t t = 0; t < ref_nb_; ++t){
			d = Math::distance2(query, &ref_[t * stride_], stride_);

			// accident happens when 2 pair points' distance are the same.
			while (dist_index_map.count(d) != 0){
				d += 0.0000000000001;
			}
			dist_index_map.insert(std::pair<double, index_t>(d, t));
		}

		index_t i = 0;
		for (
			std::map<double, index_t>::iterator iter = dist_index_map.begin();
			iter != dist_index_map.end() && i < neighbors_nb;
			++iter
			){
			dists[i] = iter->first;
			neighbors[i] = iter->second;
			i++;
		}
	}

	void Points_nn::get_nearest_neighbors(const double* query, index_t neighbors_nb, index_t offset){
		if (!selfstore_){
			fprintf(stderr, "you cannot query nn like this in non-selfstore mode");
			return;
		}
		get_nearest_neighbors(query, index_ + offset * k_, neighbors_nb, dist2_ + offset * k_);
	}
}