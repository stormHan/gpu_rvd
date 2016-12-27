/*
 * Math basic manipulation utilities
 *
 *
 */

#ifndef H_BASIC_COMMON
#define H_BASIC_COMMON

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

		inline void compute_center(const double* p1, const double* p2, const double* p3, index_t dim, float* result){
			for (coords_index_t t = 0; t < dim; ++t){
				result[t] = (float)((p1[t] + p2[t] + p3[t]) / 3);
			}
		}
	}

}

#endif /* H_BASIC_COMMON */