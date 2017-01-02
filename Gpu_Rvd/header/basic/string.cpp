/**
 * \file header/basic/string.cpp
 */

#include <basic\string.h>

namespace Gpu_Rvd{

	/**
	* \brief Builds the conversion error message
	* \param[in] s the input string that could not be converted
	* \param[in] type the expected destination type
	* \return a string that contains the error message
	*/
	std::string conversion_error(
		const std::string& s, const std::string& type
		) {
		std::ostringstream out;
		out << "Conversion error: cannot convert string '"
			<< s << "' to " << type;
		return out.str();
	}
}

namespace Gpu_Rvd{

	namespace String{

		/********************************************************************/

		ConversionError::ConversionError(
			const std::string& s, const std::string& type
			) :
			std::logic_error(conversion_error(s, type)) {
		}

		const char* ConversionError::what() const throw () {
			return std::logic_error::what();
		}

	}

}