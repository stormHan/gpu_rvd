/*
 * Math basic manipulation utilities
 *
 *
 */

#ifndef H_BASIC_COMMON
#define H_BASIC_COMMON

#include <basic\common.h>
#include <basic\numeric.h>
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

		template <class T>
		inline T tri_area(const T* p1, const T* p2, const T* p3, index_t dim){
			vec3g<T> p(p1), q(p2), r(p3);
			return length(cross(q - p, r - p)) / T(2);
		}

		template <class T>
		inline vec3g<T> random_point_tri(const vec3g<T>& p1, const vec3g<T>& p2, const vec3g<T>& p3){
			Real s = Numeric::random_float64();
			Real t = Numeric::random_float64();

			if (s + t > 1){
				s = 1 - s;
				t = 1 - t;
			}
			return (1 - s - t)*p1 + s*p2 + t*p3;
		}
		

	}

}

#endif /* H_BASIC_COMMON */