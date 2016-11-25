
#include <mesh\mesh.h>

namespace Gpu_Rvd{
	
	Points::Points() :
		vertex_nb_(0),
		is_single_precision_(false),
		is_double_precision_(true)
	{};

	Mesh::Mesh() :
		facet_nb_(0)
	{};

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
		for (index_t i = 0; i < dim; ++i){
			vertexd_.push_back(p[i]);
		}
		vertex_nb_++;
	}

	const index_t* Mesh::get_facet(index_t t) const{
		if (t >= facet_nb_){
			fprintf(stderr, "cannot get the facet as index > real number");
			return NULL;
		}
		else return &facet_[t * dimension_];
	}

	void Mesh::add_facet(const index_t* p, index_t dim){
		if (dim != dimension_){
			fprintf(stderr, "dimension not right when adding facet");
		}
		for (index_t i = 0; i < dim; ++i){
			facet_.push_back(p[i]);
		}
		facet_nb_++;
	}
}