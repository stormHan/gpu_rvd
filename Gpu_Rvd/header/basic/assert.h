/**
 * \file header/basic/assert.h
 * \brief Assertion checking mechanism
 */

#ifndef H_BASIC_ASSERT
#define H_BASIC_ASSERT

#include <basic\common.h>


namespace Gpu_Rvd{

	/**
	 * \brief Assert termination mode
	 * \detials Defines how assertion failures should
	 * terminate the program.
	 * By default, Assertion failure throw an exception.
	 */
	enum AssertMode{
		/* Assertion failures throw an exception */
		ASSERT_THROW,
		/* Assertion failures throw call abort() */
		ASSERT_ABORT
	};

	/**
	* \brief Sets assertion mode.
	* \param[in] mode assert termination mode
	* \see AssertMode
	*/
	void  set_assert_mode(AssertMode mode);

	/**
	* \brief Returns the current assert termination mode
	*/
	AssertMode  assert_mode();

	/**
	* \brief Aborts the program
	* \details On Linux, this calls the system function abort(). On Windows,
	* abort() is more difficult to see under debugger, so this creates a
	* segmentation fault by deferencing a null pointer.
	*/
	void  geo_abort();

	/**
	* \brief Prints an assertion failure
	* \details This function is called when a boolean condition is not met.
	* It prints an error message and terminates the program according to
	* the current assert termination mode.
	* \param[in] condition_string string representation of the condition
	* \param[in] file file where the assertion failed
	* \param[in] line line where the assertion failed
	*/
	void  geo_assertion_failed(
		const std::string& condition_string,
		const std::string& file, int line
		);

	/**
	* \brief Prints a range assertion failure
	* \details This function is called when a value is out of a legal range.
	* It prints an error message and terminates the program according to
	* the current assert termination mode.
	* \param[in] value the illegal value
	* \param[in] min_value minimum allowed value
	* \param[in] max_value maximum allowed value
	* \param[in] file file where the assertion failed
	* \param[in] line line where the assertion failed
	*/
	void  geo_range_assertion_failed(
		double value, double min_value, double max_value,
		const std::string& file, int line
		);

	/**
	* \brief Prints an unreachable location failure
	* \details This function is called when execution reaches a point that it
	* should not reach. It prints an error message and terminates the
	* program according to the current assert termination mode.
	* \param[in] file file containing the unreachable location
	* \param[in] line line of the unreachable location
	*/
	void  geo_should_not_have_reached(
		const std::string& file, int line
		);
}

// Three levels of assert:
// use geo_assert() and geo_range_assert()               non-expensive asserts
// use geo_debug_assert() and geo_debug_range_assert()   expensive asserts
// use geo_parano_assert() and geo_parano_range_assert() very exensive asserts

/**
* \brief Verifies that a condition is met
* \details Checks if the condition \p x. If the condition is false, it prints
* an error messages and terminates the program.
* \param[in] x the boolean expression of the condition
* \see geo_assertion_failed()
*/
#define geo_assert(x) {                                      \
        if(!(x)) {                                               \
            Gpu_Rvd::geo_assertion_failed(#x, __FILE__, __LINE__);   \
		        }                                                \
}

/**
* \brief Verifies that a value is in a legal range
* \details Verifies that value \p x is in the range [\p min_value, \p
* max_value]. If this is false, it prints an error messages and terminates
* the program.
* \param[in] x the value to verify
* \param[in] min_val minimum allowed value
* \param[in] max_val maximum allowed value
* \see geo_range_assertion_failed()
*/
#define geo_range_assert(x, min_val, max_val) {              \
        if(((x) < (min_val)) || ((x) > (max_val))) {             \
            Gpu_Rvd::geo_range_assertion_failed(x, min_val, max_val, \
                __FILE__, __LINE__                               \
            );                                                   \
		        }                                                        \
}

/**
* \brief Sets a non reachable point in the program
* \details
*/
#define geo_assert_not_reached {                             \
        Gpu_Rvd::geo_should_not_have_reached(__FILE__, __LINE__);    \
}

/**
* \def geo_debug_assert(x)
* \copydoc geo_assert()
* \note This assertion check is only active in debug mode.
*/
/**
* \def geo_debug_range_assert(x, min_val, max_val)
* \copydoc geo_range_assert()
* \note This assertion check is only active in debug mode.
*/
#ifdef GEO_DEBUG
#define geo_debug_assert(x) geo_assert(x)
#define geo_debug_range_assert(x, min_val, max_val) geo_range_assert(x, min_val, max_val)
#else
#define geo_debug_assert(x)
#define geo_debug_range_assert(x, min_val, max_val)
#endif

/**
* \def geo_parano_assert(x)
* \copydoc geo_assert()
* \note This assertion check is only active in paranoid mode.
*/
/**
* \def geo_parano_range_assert(x, min_val, max_val)
* \copydoc geo_range_assert()
* \note This assertion check is only active in paranoid mode.
*/
#ifdef GEO_PARANOID
#define geo_parano_assert(x) geo_assert(x)
#define geo_parano_range_assert(x, min_val, max_val) geo_range_assert(x, min_val, max_val)
#else
#define geo_parano_assert(x)
#define geo_parano_range_assert(x, min_val, max_val)
#endif

#endif /* H_BASIC_ASSERT */