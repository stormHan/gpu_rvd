
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

		virtual bool stride_supported() const { return true; }
		
		virtual void set_points(
			index_t nb_points, const double* points, index_t stride
			);

		virtual void get_nearest_neighbors(
			index_t nb_neighbors,
			const double* query_point,
			index_t* neighbors,
			double* neighbors_sq_dist
			) const;

		virtual void get_nearest_neighbors(
			index_t nb_neighbors,
			index_t query_point,
			index_t* neighbors,
			double* neighbors_sq_dist
			) const;

		
	public:
		/**
		* \brief Used by multithread tree construction
		* in the implementation of set_points()
		*/
		void operator() (index_t t);

		//virtual ~Kdtree();
	protected:
		virtual ~Kdtree();

		/*
		 * \brief Number of points stored in the leafs of the tree.
		 */
		static const index_t MAX_LEAF_SIZE = 16;

		/*
		 * \brief Returns the maximum node index in subtree
		 * \param[in] node_id node index of the subtree
		 * \param[in] b first index of the points sequence in subtree.
		 * \param[in] e one position past the last index of the points
		 * sequence in the subtree.
		 */
		static index_t max_node_index(
			index_t node_id, index_t b, index_t e
			){
			//when recursive to leaf node, return 
			//the index of node_id
			if (e - b <= MAX_LEAF_SIZE){
				return node_id;
			}

			index_t m = b + (e - b) / 2;
			return geo_max(
				max_node_index(node_id * 2, b, m),
				max_node_index(node_id * 2 + 1, m, e)
				);
		}

		/**
		* \brief The context for traversing a KdTree.
		* \details Stores a sorted sequence of (point,distance)
		*  couples.
		*/
		struct NearestNeighbors{

			/**
			* \brief Creates a new NearestNeighbors
			* \details Storage is provided
			* and managed by the caller.
			* Initializes neighbors_sq_dist[0..nb_neigh-1]
			* to Numeric::max_float64() and neighbors[0..nb_neigh-1]
			* to index_t(-1).
			* \param[in] nb_neighbors_in number of neighbors to retreive
			* \param[in] neighbors_in storage for the neighbors, allocated
			*  and managed by caller
			* \param[in] neighbors_sq_dist_in storage for neighbors squared
			*  distance, allocated and managed by caller
			*/
			NearestNeighbors(
				index_t nb_neighbors_in,
				index_t* neighbors_in,
				double* neighbors_sq_dist_in
				) :
				nb_neighbors(nb_neighbors_in),
				neighbors(neighbors_in),
				neighbors_sq_dist(neighbors_sq_dist_in)
			{
				for (index_t t = 0; t < nb_neighbors; ++t){
					neighbors[t] = index_t(-1);
					neighbors_sq_dist[t] = Numeric::max_float64();
				}
			}

			/**
			 * \brief Inserts a new neighbor.
			 * \details Only the nb_neighbor nearest points are kept.
			 * \param[in] neighbor the index of the point.
			 * \param[in] sq_dist the squared distance between the point
			 * and the query point.
			 */
			void insert(
				index_t neighbor, double sq_dist
				){
				if (sq_dist >= furthest_neighbor_sq_dist())
					return;
				index_t i = nb_neighbors;
				while (i != 0 && neighbors_sq_dist[i - 1] > sq_dist){
					if (i < nb_neighbors){
						neighbors[i] = neighbors[i - 1];
						neighbors_sq_dist[i] = neighbors_sq_dist[i - 1];
					}
					i--;
				}
				neighbors[i] = neighbor;
				neighbors_sq_dist[i] = sq_dist;
			}

			/**
			 * \brief Gets the squared distance to the furthest neighbor.
			 */
			double furthest_neighbor_sq_dist() const{
				return neighbors_sq_dist[nb_neighbors - 1];
			}

			index_t nb_neighbors;
			index_t* neighbors;
			double* neighbors_sq_dist;
		};

		/**
		 * \brief Computes the coordinate along which a point
		 *	sequence will be splitted
		 */
		coords_index_t best_splitting_coord(index_t b, index_t e);

		/**
		* \brief Computes the extent of a point sequence along a coordinate.
		* \param[in] b first index of the point sequence
		* \param[in] e one position past the last index of the point sequence
		* \param[in] coord coordinate along which the extent is measured
		*/
		double spread(
			index_t b, index_t e, coords_index_t coord
			) {
			double minval = Numeric::max_float64();
			double maxval = Numeric::min_float64();
			for (index_t i = b; i < e; ++i) {
				double val = point_ptr(point_index_[i])[coord];
				minval = geo_min(minval, val);
				maxval = geo_max(maxval, val);
			}
			return maxval - minval;
		}

		/**
		* \brief Creates the subtree under a node.
		* \param[in] node_index index of the node that represents
		*  the subtree to create
		* \param[in] b first index of the point sequence in the subtree
		* \param[in] e one position past the last index of the point
		*  index in the subtree
		*/
		void create_kd_tree_recursive(
			index_t node_index, index_t b, index_t e
			){
			if (e - b <= MAX_LEAF_SIZE){
				return;
			}
			index_t m = split_kd_node(node_index, b, e);
			create_kd_tree_recursive(node_index * 2, b, m);
			create_kd_tree_recursive(node_index * 2 + 1, m, e);
		}

		/**
		 * \brief Computes and stores the splitting coordinate
		 * and splitting value of the node node_index, that
		 * correspponds to the [b, e) points sequence.
		 *
		 * \return a node index m. The point sequence [b, m)
		 * and [m, e)correspond to the left child(2*node_index)
		 * and right child (2*node_index+1) of node_index
		 */
		index_t split_kd_node(
			index_t node_index, index_t b, index_t e
			);

		/**
		* \brief The recursive function to implement KdTree traversal and
		*  nearest neighbors computation.
		* \details Traverses the subtree under the
		*  node_index node that corresponds to the
		*  [b,e) point sequence. Nearest neighbors
		*  are inserted into neighbors during
		*  traversal.
		* \param[in] node_index index of the current node in the Kd tree
		* \param[in] b index of the first point in the subtree under
		*  node \p node_index
		* \param[in] e one position past the index of the last point in the
		*  subtree under node \p node_index
		* \param[in,out] bbox_min coordinates of the lower
		*  corner of the bounding box.
		*  Allocated and managed by caller.
		*  Modified by the function and restored on exit.
		* \param[in,out] bbox_max coordinates of the
		*  upper corner of the bounding box.
		*  Allocated and managed by caller.
		*  Modified by the function and restored on exit.
		* \param[in] bbox_dist squared distance between
		*  the query point and a bounding box of the
		*  [b,e) point sequence. It is used to early
		*  prune traversals that do not generate nearest
		*  neighbors.
		* \param[in] query_point the query point
		* \param[in,out] neighbors the computed nearest neighbors
		*/
		void get_nearest_neighbors_recursive(
			index_t node_index, index_t b, index_t e,
			double* bbox_min, double* bbox_max,
			double bbox_dist, const double* query_point,
			NearestNeighbors& neighbors
			) const;

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