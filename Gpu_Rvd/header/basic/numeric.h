/*
	handle basic operation about number.
*/

#ifndef BASIC_NUMERIC_H
#define BASIC_NUMERIC_H

#include <basic\common.h>
#include <basic\vec.h>
#include <stdint.h>
// indice of loop
//typedef unsigned int index_t

namespace Gpu_Rvd{

	/**
	* A namespace gathering typedefs
	* corresponding to numbers. These types
	* names have the form (u)int<size> or float<size>,
	* where the (optional) u denotes an unsigned type,
	* and the size is in bits.
	*/

	namespace Numeric {
		/** Generic pointer type */
		typedef void* pointer;

		/** Integer type with a width of 8 bits */
		typedef int8_t int8;

		/** Integer type with a width of 16 bits */
		typedef int16_t int16;

		/** Integer type with a width of 32 bits */
		typedef int32_t int32;

		/** Integer type with a width of 64 bits */
		typedef int64_t int64;

		/** Unsigned integer type with a width of 8 bits */
		typedef uint8_t uint8;

		/** Unsigned integer type with a width of 16 bits */
		typedef uint16_t uint16;

		/** Unsigned integer type with a width of 32 bits */
		typedef uint32_t uint32;

		/** Unsigned integer type with a width of 64 bits */
		typedef uint64_t uint64;

		/** Floating point type with a width of 32 bits */
		typedef float float32;

		/** Floating point type with a width of 64 bits */
		typedef double float64;
		
		int32 random_int32();

		float64 random_float64();

		float32 random_float32();

		bool  is_nan(float32 x);

		bool  is_nan(float64 x);
	}
	
	typedef Numeric::float64 Real;
	typedef Numeric::uint32 index_t;

	typedef vec3g<Real> vec3;
	typedef vec2g<Real> vec2;

	enum Sign{
		NEGATIVE = -1,
		ZERO = 0,
		POSITIVE = 1
	};

	template <class T>
	inline Sign geo_sgn(const T& x){
		return (x > 0) ? POSITIVE : (
			(x < 0) ? NEGATIVE : ZERO;
		);
	}

	template <class T>
	inline T geo_abs(T x){
		return (x >= 0) ? x : -x;
	}

	inline index_t max_index_t() {
		return (std::numeric_limits<index_t>::max)();
	}

	inline index_t min_index_t() {
		return (std::numeric_limits<index_t>::min)();
	}
}
#endif /* BASIC_NUMERIC_H */
