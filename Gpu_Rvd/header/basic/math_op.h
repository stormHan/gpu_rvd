/*
 * Math basic manipulation utilities
 *
 *
 */

#include <basic\common.h>

#include <math.h>

namespace Gpu_Rvd{

	namespace Math{

		/*
		 * \brief compute the square of the distance of 2 points.
		 */
		template <typename T>
		inline T distance2(const T* p1, const T* p2, coords_index_t dim){
			T result = 0;
			for (coords_index_t c = 0; c < dim; ++c){
				result += (p1[c] - p2[c]) * (p1[c] - p2[c]);
			}
			return result;
		}
	}

}