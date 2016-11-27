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

}
#endif /* BASIC_COMMON_H */