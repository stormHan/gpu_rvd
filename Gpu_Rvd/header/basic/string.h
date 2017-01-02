/*
	implementation of String operation
*/

#ifndef BASIC_STRING_H
#define BASIC_STRING_H

#include <basic\common.h>
#include <basic\numeric.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <limits.h>
namespace Gpu_Rvd{

	

	/*
	 *\breif String manipulation utilities
	 */
	namespace String{
		using namespace Numeric;

		template <typename T>
		inline std::string to_string(const T& value){
			std::ostringstream out;
			out << value;
			return out.str();
		}

		template <class T>
		inline bool from_string(const char* s, T& value) {
			std::istringstream in(s);
			return (in >> value >> std::ws) && in.eof();
		}

		/**
		* \brief Conversion exception
		* \details This exception is thrown by the conversion functions
		* to_bool(), to_int() and to_double() when a string cannot be
		* converted to the desired type.
		*/
		class  ConversionError : public std::logic_error {
		public:
			/**
			* \brief Constructs a conversion exception
			* \param[in] s the input string that could not be converted
			* \param[in] type the expected destination type
			*/
			ConversionError::ConversionError(
				const std::string& s, const std::string& type
				);
			/**
			* \brief Gets the string identifying the exception
			*/
			const char* ConversionError::what() const throw ();
		};

		/**
		* \brief Converts a std::string to a typed value
		* \details This is a generic version that uses a std::istringstream
		* to extract the value from the string. This function is specialized
		* for integral types to reach the maximum efficiency.
		* \param[in] s the source string
		* \param[out] value the typed value
		* \retval true if the conversion was successful
		* \retval false otherwise
		*/
		template<typename T>
		inline bool from_string(const std::string& s, T& value){
			return from_string(s.c_str(), value);
		}

		/**
		* \brief Converts a string to a double value
		* \param[in] s the source string
		* \param[out] value the double value
		* \retval true if the conversion was successful
		* \retval false otherwise
		*/
		//template <>
		inline bool from_string(const char* s, double& value){
			char* end;
			value = strtod(s, &end);
			return end != s && *end == '\0';
		}

		/**
		* \brief Converts a string to a signed integer value
		* \param[in] s the source string
		* \param[out] value the integer value
		* \retval true if the conversion was successful
		* \retval false otherwise
		*/
		template<typename T>
		inline bool string_to_signed_integer(const char* s, T& value){
			char* end;
			int64 v = _strtoi64(s, &end, 10);

			if (
				end != s && *end == '\0' &&
				v >= (std::numeric_limits<T>::min)() &&
				v <= (std::numeric_limits<T>::max)()
				){
				value = static_cast<T>(v);
				return true;
			}
			return false;
		}

		/**
		* \brief Converts a string to a Numeric::int8 value
		* \see string_to_signed_integer()
		*/
		
		inline bool from_string(const char* s, int8& value) {
			return string_to_signed_integer(s, value);
		}

		/**
		* \brief Converts a string to a Numeric::int16 value
		* \see string_to_signed_integer()
		*/
		
		inline bool from_string(const char* s, int16& value) {
			return string_to_signed_integer(s, value);
		}

		/**
		* \brief Converts a string to a Numeric::int32 value
		* \see string_to_signed_integer()
		*/
		
		inline bool from_string(const char* s, int32& value) {
			return string_to_signed_integer(s, value);
		}

		inline bool from_string(const char* s, int64& value){
			return string_to_signed_integer(s, value);
		}

		template<typename T>
		inline bool string_to_unsigned_integer(const char* s, T& value){
			uint64 v = 0;
			char* end;
			v = _strtoi64(s, &end, 10);
			
			if (
				end != s && *end == '\0' &&
				v >= (std::numeric_limits<T>::min)() &&
				v <= (std::numeric_limits<T>::max)()
				){
				value = static_cast<T>(v);
				return true;
			}
			return false;
		}

		/**
		* \brief Converts a string to a Numeric::uint8 value
		* \see string_to_unsigned_integer()
		*/
		template <>
		inline bool from_string(const char* s, uint8& value) {
			return string_to_unsigned_integer(s, value);
		}

		/**
		* \brief Converts a string to a Numeric::uint16 value
		* \see string_to_unsigned_integer()
		*/
		template <>
		inline bool from_string(const char* s, uint16& value) {
			return string_to_unsigned_integer(s, value);
		}

		/**
		* \brief Converts a string to a Numeric::uint32 value
		* \see string_to_unsigned_integer()
		*/
		template <>
		inline bool from_string(const char* s, uint32& value) {
			return string_to_unsigned_integer(s, value);
		}

		/**
		* \brief Converts a string to a Numeric::uint64 value
		*/
		template <>
		inline bool from_string(const char* s, uint64& value){
			char* end;
			value = _strtoui64(s, &end, 10);
			return end != s && *end == '\0';
		}

		/**
		* \brief Converts a string to a boolean value
		* \details
		* Legal values for the true boolean value are "true","True" and "1".
		* Legal values for the false boolean value are "false","False" and "0".
		* \param[in] s the source string
		* \param[out] value the boolean value
		* \retval true if the conversion was successful
		* \retval false otherwise
		*/
		template <>
		inline bool from_string(const char* s, bool& value) {
			if (strcmp(s, "true") == 0 ||
				strcmp(s, "True") == 0 ||
				strcmp(s, "1") == 0
				) {
				value = true;
				return true;
			}
			if (strcmp(s, "false") == 0 ||
				strcmp(s, "False") == 0 ||
				strcmp(s, "0") == 0
				) {
				value = false;
				return true;
			}
			return false;
		}

		/**
		* \brief Converts a string to an int
		* \details If the entire string cannot be
		* converted to an int, the function
		* throws an exception ConversionError.
		* \param[in] s the source string
		* \return the extracted integer value
		* \see ConversionError
		*/
		inline int to_int(const std::string& s) {
			int value;
			if (!from_string(s, value)) {
				throw ConversionError(s, "integer");
			}
			return value;
		}

		/**
		* \brief Converts a string to an unsigned int
		* \details If the entire string cannot be
		* converted to an unsigned int, the function
		* throws an exception ConversionError.
		* \param[in] s the source string
		* \return the extracted integer value
		* \see ConversionError
		*/
		inline unsigned int to_uint(const std::string& s) {
			unsigned int value;
			if (!from_string(s, value)) {
				throw ConversionError(s, "integer");
			}
			return value;
		}

		/**
		* \brief Converts a string to a double
		* \details If the entire string cannot be
		* converted to a double, the function
		* throws an exception ConversionError.
		* \param[in] s the source string
		* \return the extracted double value
		* \see ConversionError
		*/
		inline double to_double(const std::string& s) {
			double value;
			if (!from_string(s, value)) {
				throw ConversionError(s, "double");
			}
			return value;
		}

		/**
		* \brief Converts a string to a boolean
		* \details If the entire string cannot be
		* converted to a boolean, the function
		* throws an exception ConversionError.
		* \param[in] s the source string
		* \return the extracted boolean value
		* \see ConversionError
		*/
		inline bool to_bool(const std::string& s) {
			bool value;
			if (!from_string(s, value)) {
				throw ConversionError(s, "boolean");
			}
			return value;
		}
	}
}

#endif /* BASIC_STRING_H */