/************************************
	datastruct Mesh

	Store the infomartion of Model mesh.

************************************/
#ifndef MESH_MESH_H
#define MESH_MESH_H

#include <basic\common.h>

namespace Gpu_Rvd{
	
	class Points{
	public:
		Points();

		const index_t dimension() const{ return dimension_; }

		bool is_single_precision();

		bool is_double_precision();

		const double* get_vertexd(index_t t) const;

		void add_vertexd(const double* p, index_t dim);

		void add_vertexd(const std::vector<double>& p);

		const index_t get_vertex_nb() const{ return vertex_nb_; }

	protected:
		std::vector<double>		vertexd_;
		std::vector<float>		vertexf_;
		index_t					vertex_nb_;

		const static index_t	dimension_ = 3;
		bool					is_single_precision_;
		bool					is_double_precision_;
	};


	class Mesh : public Points{
	public:
		Mesh();

		Mesh(const Mesh& M);

		const index_t* get_facet(index_t t) const;

		void add_facet(const index_t* p, index_t dim);

		void add_facet(const std::vector<index_t>& p);

		const index_t get_facet_nb() const{ return facet_nb_; }


	private:
		
		std::vector<index_t>	facet_;
		index_t					facet_nb_;

		
		
	};

}

#endif /* MESH_MESH_H */