/**
 * \file header/basic/assert.cpp
 * \brief implementation of assert.h
 */

#include <basic\assert.h>
#include <stdlib.h>
#include <sstream>
#include <stdexcept>

namespace Gpu_Rvd{

	namespace {
		AssertMode assert_mode_ = ASSERT_THROW;
		bool aborting_ = false;
	}

	void set_assert_mode(AssertMode mode) {
		assert_mode_ = mode;
	}

	AssertMode assert_mode() {
		return assert_mode_;
	}

	void geo_abort(){
		//Avoid assert in assert !!
		if (aborting_){

		}
		aborting_ = true;
		abort();
	}

	void geo_assertion_failed(
		const std::string& condition_string,
		const std::string& file, int line
		) {
		std::ostringstream os;
		os << "Assertion failed: " << condition_string << ".\n";
		os << "File: " << file << ",\n";
		os << "Line: " << line;

		if (assert_mode_ == ASSERT_THROW) {
			throw std::runtime_error(os.str());
		}
		else {
			std::cerr <<"Assert " << os.str() << std::endl;
			geo_abort();
		}
	}

	void geo_range_assertion_failed(
		double value, double min_value, double max_value,
		const std::string& file, int line
		) {
		std::ostringstream os;
		os << "Range assertion failed: " << value
			<< " in [ " << min_value << " ... " << max_value << " ].\n";
		os << "File: " << file << ",\n";
		os << "Line: " << line;

		if (assert_mode_ == ASSERT_THROW) {
			throw std::runtime_error(os.str());
		}
		else {
			std::cerr << "Assert " << os.str() << std::endl;
			geo_abort();
		}
	}

	void geo_should_not_have_reached(
		const std::string& file, int line
		) {
		std::ostringstream os;
		os << "Control should not have reached this point.\n";
		os << "File: " << file << ",\n";
		os << "Line: " << line;

		if (assert_mode_ == ASSERT_THROW) {
			throw std::runtime_error(os.str());
		}
		else {
			std::cerr << "Assert " << os.str() << std::endl;
			geo_abort();
		}
	}
}