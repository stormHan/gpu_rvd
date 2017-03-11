/************************************
	datastruct Mesh

	Store the infomartion of Model mesh.

************************************/
#ifndef MESH_MESH_H
#define MESH_MESH_H

#include <basic\common.h>
#include <basic\math_op.h>
#include <mesh\mesh_nn.h>

namespace Gpu_Rvd{
	
	class Points{
	public:
		Points();

		/*
		 * \brief Constructs the points in mesh
		 */
		//Points(Mesh m);

		virtual ~Points(){}
		/*
		 * \brief Gets the dimension fo the Points.
		 */
		const index_t dimension() const{
			return dimension_; 
		}

		/*
		 * \brief clear the Points
		 */
		void clear() {
			vertexf_.resize(0);
			vertexd_.resize(0);
			vertex_nb_ = 0;
			points_nn_.clear();
		}

		/*
		 * \brief Checks if the data is single precision.
		 */
		bool is_single_precision();

		/*
		* \brief Checks if the data is double precision.
		*/
		bool is_double_precision();
		
		/*
		* \brief Get the vertex(double) with the indice.
		*/
		const double* get_vertexd(index_t t) const;

		/*
		* \brief Adds a vertex(double) which data is in pointer.
		*/
		void add_vertexd(const double* p, index_t dim);

		/*
		* \brief Adds a vertex(double) which data is std::vector<double>.
		*/
		void add_vertexd(const std::vector<double>& p);

		/*
		 * \brief Adds a vertex(double) which data is in vec3
		 */
		void add_vertexd(const vec3& p);

		/*
		 * \brief Sets the vertex value by index
		 */
		void set_vertex(const double* value, index_t dimension, index_t v);

		/*
		 * \breif Gets the vertex number.
		 */
		const index_t get_vertex_nb() const{
			return vertex_nb_; 
		}

		/*
		 * \brief Gets the beginning pointer of the vertex.
		 */
		const double* v_ptr(){
			return &vertexd_[0];
		}

		/*
		 * \breif Initializes of the Points_nn.
		 */
		void initialize_points_nn(index_t k, coords_index_t dim, bool selfstore);

		/*
		 * \brief Searches the nearest neighbors
		 * \detials \param[double* query]	the query data.
		 *			\param[dim]				indicates the stride of the data of one single point.
		 *			\param[search_nb] 		indicates how many points to be searched.
		 *			\param[ind]				index_t type pointer to store the search result.
		 *			\param[dist]			double type pointer to store the distance2 result.
		 */
		void search_nn(const double* query, coords_index_t dim, index_t search_nb, index_t* ind, double* dist, index_t k);

		/**
		 * \brief Searches the nearest neighbors in self-store mode.
		 */
		void search_nn(const double* query, coords_index_t dim, index_t search_nb, index_t k);


		/*
		 * \brief Gets the points_nn's indices
		 */
		const index_t* get_indices() const{
			return points_nn_.get_index();
		}

		/*
		 * \brief Gets the k which represents the neighbor size.
		 */
		const index_t get_k() const{
			return points_nn_.get_k();
		}

	protected:
		std::vector<double>		vertexd_;
		std::vector<float>		vertexf_;
		index_t					vertex_nb_;

		const static index_t	dimension_ = 3;
		bool					is_single_precision_;
		bool					is_double_precision_;

		Points_nn				points_nn_;

	};


	class Mesh : public Points{
	public:
		Mesh();

		//Mesh(const Mesh& M);

		~Mesh(){}

		const index_t* get_facet(index_t t) const;

		void add_facet(const index_t* p, index_t dim);

		void add_facet(const std::vector<index_t>& p);

		const index_t get_facet_nb() const{ return facet_nb_; }

		const index_t* f_ptr() const{
			return &facet_[0];
		}

		/*
		 * \brief Sample random points in triangle surface.
		 */
		void init_samples(Points& p, index_t nb, std::vector<int>& sample_facet);
	private:
		
		std::vector<index_t>	facet_;
		index_t					facet_nb_;

	};

}

#endif /* MESH_MESH_H */