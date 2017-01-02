/**
* \file geogram/basic/argused.h
* \brief A function to suppress unused parameters compilation warnings
*/

#ifndef __GEOGRAM_BASIC_ARGUSED__
#define __GEOGRAM_BASIC_ARGUSED__

#include <basic/common.h>

namespace Gpu_Rvd {

	/**
	* \brief Suppresses compiler warnings about unused parameters
	* \details This function is meant to get rid of warnings
	* concerning non used parameters (for instance,
	* parameters to callbacks). The corresponding code
	* is supposed to be wiped out by the optimizer.
	*/
	template <class T>
	inline void geo_argused(const T&) {
	}
}

#endif