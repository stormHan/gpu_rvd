
#include <mesh\mesh.h>

namespace Gpu_Rvd{
	
	Points::Points() :
		vertex_nb_(0),
		is_single_precision_(false),
		is_double_precision_(true)
	{};

	Mesh::Mesh() :
		facet_nb_(0)
	{
	};

	bool Points::is_double_precision(){
		if (is_double_precision_) return true;
		else return false;
	}

	bool Points::is_single_precision(){
		if (is_single_precision_) return true;
		else return false;
	}

	const double* Points::get_vertexd(index_t t) const {
		if (t >= vertex_nb_ || !is_double_precision_){
			fprintf(stderr, "cannot get the vertex as index > real number");
			return NULL;
		}
		else return &vertexd_[t * dimension_];
	}

	void Points::add_vertexd(const double* p, index_t dim){
		if (dim != dimension_){
			fprintf(stderr, "dimension not right when adding vertex");
			return;
		}
		for (coords_index_t i = 0; i < dim; ++i){
			vertexd_.push_back(p[i]);
		}
		vertex_nb_++;
	}

	void Points::add_vertexd(const std::vector<double>& p){
		if (p.size() != dimension_){
			fprintf(stderr, "dimension not right when adding vertex");
			return;
		}
		for (coords_index_t c = 0; c < dimension_; ++c){
			vertexd_.push_back(p[c]);
		}
		vertex_nb_++;
	}

	void Points::initialize_points_nn(index_t k, coords_index_t dim, bool selfstore){
		points_nn_.init(k, dim, selfstore);
		points_nn_.set_points(vertex_nb_, v_ptr());
	}

	void Points::search_nn(const double* query, coords_index_t dim, index_t search_nb, index_t* ind, double* dist, index_t k){
		if (dim != dimension_){
			fprintf(stderr, "the dimension of the query data differs from the points'");
		}
		initialize_points_nn(k, dimension_, false);
		index_t neighbors_nb = k;
		if (neighbors_nb > 0 && neighbors_nb < vertex_nb_){
			for (index_t t = 0; t < search_nb; ++t){
				points_nn_.get_nearest_neighbors(query + dim * t,
					ind + neighbors_nb * t,
					neighbors_nb,
					dist + neighbors_nb * t);
			}
		}
	}

	void Points::search_nn(const double* query, coords_index_t dim, index_t search_nb, index_t k){
		if (dim != dimension_){
			fprintf(stderr, "the dimension of the query data differs from the points'");
		}
		points_nn_.clear();
		initialize_points_nn(k, dimension_, true);
		if (points_nn_.malloc_dist2(search_nb) &&
			points_nn_.malloc_index(search_nb)){
			index_t neighbors_nb = k;
			if (neighbors_nb > 0 && neighbors_nb < vertex_nb_){
				for (index_t t = 0; t < search_nb; ++t){
					printf(" ---doing %d---\n", t);
					points_nn_.get_nearest_neighbors(query + dim * t,
						neighbors_nb, t);
				}
			}
		}
	}

	const index_t* Mesh::get_facet(index_t t) const{
		if (t >= facet_nb_){
			fprintf(stderr, "cannot get the facet as index > real number");
			return nil;
		}
		else return &facet_[t * dimension_];
	}

	void Mesh::add_facet(const index_t* p, index_t dim){
		if (dim != dimension_){
			fprintf(stderr, "dimension not right when adding facet");
			return;
		}
		for (index_t i = 0; i < dim; ++i){
			facet_.push_back(p[i]);
		}
		facet_nb_++;
	}

	void Mesh::add_facet(const std::vector<index_t>& p){
		if (p.size() != dimension_){
			fprintf(stderr, "dimension not right when adding facet");
			return;
		}
		for (coords_index_t c = 0; c < dimension_; ++c){
			facet_.push_back(p[c]);
		}
		facet_nb_++;
	}
}