
#ifndef H_POINTS_NNSEARCH
#define H_POINTS_NNSEARCH

#include <basic\common.h>
#include <basic\numeric.h>
#include <basic\smart_pointer.h>
#include <basic\counted.h>
/*
 * \file header/points/nn_search.h
 * \brief Abstract interface for nearest neighbor seaching
 */

namespace Gpu_Rvd{

	class NearestNeighborSearch : public Counted{
	public:
		/**
		* \brief Creates a new search algorithm
		* \param[in] dimension dimension of the points (e.g., 3 for 3d)
		* \param[in] name name of the search algorithm to create:
		* - "ANN" - uses the standard ANN algorithm
		* - "BNN" - uses the optimized KdTree
		* - "default", uses the command line argument "algo:nn_search"
		* \retval nil if \p name is not a valid search algorithm name
		* \retval otherwise, a pointer to a search algorithm object. The
		* returned pointer must be stored in a NearestNeighborSearch_var that
		* does automatic destruction:
		* \code
		* NearestNeighborSearch_var nnsearch =
		*     NearestNeighborSearch::create(3, "ANN");
		* \endcode
		*/
		static NearestNeighborSearch* create(
			coords_index_t dimension, const std::string& name = "default"
			);

		virtual void get_nearest_neighbors(
			index_t nb_neighbors,
			const double* query_point,
			index_t* neighbors,
			double* neighbors_sq_dist
			) const = 0;

		/**
		* \brief Finds the nearest neighbors of a point given by
		*  its index.
		* \details For some implementation, may be faster than
		*  nearest neighbor search by point coordinates.
		* \param[in] nb_neighbors number of neighbors to be searched.
		*  Should be smaller or equal to nb_points() (else it triggers
		*  an assertion)
		* \param[in] query_point as the index of one of the points that
		*  was inserted in this NearestNeighborSearch
		* \param[out] neighbors array of nb_neighbors index_t
		* \param[out] neighbors_sq_dist array of nb_neighbors doubles
		*/
		virtual void get_nearest_neighbors(
			index_t nb_neighbors,
			index_t query_point,
			index_t* neighbors,
			double* neighbors_sq_dist
			) const;

		/**
		* \brief Nearest neighbor search.
		* \param[in] query_point array of dimension() doubles
		* \return the index of the nearest neighbor from \p query_point
		*/

		index_t get_nearest_neighbor(
			const double* query_point
			) const {
			index_t result;
			double sq_dist;
			get_nearest_neighbors(1, query_point, &result, &sq_dist);
			return index_t(result);
		}

		/**
		* \brief Sets the points and create the search data structure.
		* \param[in] nb_points number of points
		* \param[in] points an array of nb_points * dimension()
		*/ 
		virtual void set_points(index_t nb_points, const double* points);

		/**
		* \brief Gets the dimension of the points.
		* \return the dimension
		*/
		coords_index_t dimension() const {
			return dimension_;
		}
		/**
		* \brief Gets the number of points.
		* \return the number of points
		*/
		index_t nb_points() const {
			return nb_points_;
		}

		/**
		* \brief Gets a point by its index
		* \param[in] i index of the point
		* \return a const pointer to the coordinates of the point
		*/
		const double* point_ptr(index_t i) const {
			return points_ + i * stride_;
		}

		/**
		* \brief Search can be exact or approximate. Approximate
		*  search may be faster.
		* \return true if nearest neighbor search is exact,
		*   false otherwise
		*/
		bool exact() const {
			return exact_;
		}

		virtual void set_exact(bool x){
			exact_ = x;
		}

		/**
		* \brief Tests whether the stride variant of set_points() is supported
		* \return true if stride different from dimension can be used
		*  in set_points(), false otherwise
		*/
		virtual bool stride_supported() const;

	protected: 
		NearestNeighborSearch(coords_index_t dimension);

		virtual ~NearestNeighborSearch();

	protected:
		coords_index_t dimension_;
		index_t nb_points_;
		index_t stride_;
		const double* points_;
		bool exact_;
	};

	/**
	* \brief A smart pointer that contains a NearestNeighborSearch object.
	* \relates NearestNeighborSearch
	*/
	typedef SmartPointer<NearestNeighborSearch> NearestNeighborSearch_var;
}


#endif /* H_POINTS_NNSEARCH */