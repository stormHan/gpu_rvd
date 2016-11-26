/*
	handle basic operation about number.
*/

#ifndef BASIC_NUMERIC_H
#define BASIC_NUMERIC_H

#include <basic\common.h>

#include <stdint.h>
// indice of loop
//typedef unsigned int index_t

namespace Gpu_Rvd{

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

}
#endif /* BASIC_NUMERIC_H */
