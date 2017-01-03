/*
	do the prework
*/

#ifndef BASIC_COMMON_H
#define BASIC_COMMON_H

/*
	this header should be included before anything else
*/
#include <iostream>

/*
	include the STL container so that we can be more comfortable 
	when we handle something like the file.
*/
#include <vector>
#include <string>

namespace Gpu_Rvd{

	/* Represents the non-negative indices */
	typedef unsigned int index_t;

	/* Represents the possibly negative indices */
	typedef int signed_index_t;

	/* Represents the small size coords indices */
	typedef unsigned char coords_index_t;

#define nil 0

#ifdef NDEBUG
#undef GEO_DEBUG
#undef GEO_PARANOID
#else
#define GEO_DEBUG
#define GEO_PARANOID
#endif

#if defined(WIN32) || defined(_WIN64)

#define GEO_OS_WINDOWS

#if defined(_MSC_VER)
# define GEO_COMPILER_MSVC
#else
# error "Unsupported compiler"
#endif

#if defined(_WIN64)
#  define GEO_ARCH_64
#else
#  define GEO_ARCH_32
#endif

#endif
}
#endif /* BASIC_COMMON_H */