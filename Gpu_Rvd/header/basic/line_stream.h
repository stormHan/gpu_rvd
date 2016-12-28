/*
	Utility class to read a file line per line and parsing
	fields from each line
*/

#include <basic\common.h>
#include <basic\string.h>

namespace Gpu_Rvd{

	using namespace Numeric;

	class LineInput{
	public:
		/*
			Creates a new line reader from a file
			This open the file "filename" for reading and prepares to
			read it line by line.If the file could not be opened, OK() 
			will return false.
		*/
		LineInput(const std::string& filename);

		/*
			Destory the line reader.
			this will close the File.
		*/
		~LineInput();

		/*
			check if the File is available
		*/
		bool OK() const{
			return ok_;
		}

		/*
			Check if line reader has reached the end of the input stream
		*/
		bool eof() const{
			return feof(F_) ? true : false;
		}

		/*
			Read a new line
		*/
		bool get_line();

		/**
		* \brief Gets the number of fields in the current line
		* \details Function get_fields() must be called once after get_line()
		* before calling this function, otherwise the result is undefined.
		* \return the number of fields in the current line
		*/
		index_t nb_fields() const {
			return index_t(field_.size());
		}

		/**
		* \brief Returns the current line number
		* \details If no line has been read so far, line_number() returns 0.
		*/
		size_t line_number() const {
			return line_num_;
		}

		/*
			Gets a line field as a modifiable string
			The function returns the field at index \p i.
			Function get_field() must be called once after get_line() before
			calling this function, otherwise the result is undefined.
		*/
		char* field(index_t i) {
			if (i < nb_fields()){
				return field_[i];
			}
			return nil;
		}

		/*
			Gets a line field as a non-modifiable string
			The function returns the field at index \p i.
			Function get_field() must be called once after get_line() before
			calling this function, otherwise the result is undefined.
		*/
		const char* field(index_t i) const{
			if (i < nb_fields()){
				return field_[i];
			}
			return nil;
		}

		/**
		* \brief Gets a line field as an signed integer.
		*/
		signed_index_t field_as_int(index_t i) const{
			signed_index_t result = 0;
			if (!String::from_string(field_[i], result)){
				fprintf(stderr, "failed to turn the field[%d] into integer", i);
			}
			return result;
		}

		/**
		 * \brief Gets a line field as an unsigned integer.
	 	 */
		index_t field_as_uint(index_t i) const {
			index_t result = 0;
			if (!String::from_string(field_[i], result)){
				fprintf(stderr, "failed to turn the field[%d] into non-integer", i);
			}
			return result;
		}

		/**
		 *\brief Gets a line field as a double.
		 */
		double field_as_double(index_t i) const{
			double result = 0.0;
			if (!String::from_string(field_[i], result)){
				fprintf(stderr, "failed to turn the field[%d] into double.", i);
			}
			return result;
		}

		/**
		* \brief Compares a field with a string.
		* \details The function compares the field at index \p i with string
		* \p s and returns \c true if they are equal. Function get_fields()
		* must be called once after get_line() before calling this function,
		* otherwise the result is undefined.
		* \param[in] i the index of the field
		* \param[in] s the string to compare the field to
		* \retval true if field at index \p i equals string \p s
		* \retval false otherwise
		*/
		bool field_matches(index_t i, const char* s) const {
			return strcmp(field(i), s) == 0;
		}

		/**
		* \brief Splits the current line into fields.
		* \details The function uses \p separators to split the
		* current line into individual fields that can be accessed
		* by field() and its typed variants.
		* \param[in] separators a string that contains all
		*  the characters considered as separators.
		* \see field()
		*/
		void get_fields(const char* separators = " \t\r\n");

		/**
		* \brief Gets the current line.
		* \details If get_fields() was called, then an end-of-string
		*  marker '\0' is present at the end of the first field.
		* \return a const pointer to the internal buffer that stores
		*  the current line
		*/
		const char* current_line() const {
			return line_;
		}
	private:
		static const index_t MAX_LINE_LEN = 65535;
		
		FILE* F_;
		std::string file_name_;
		size_t line_num_;
		char line_[MAX_LINE_LEN];		//current line
		std::vector<char*> field_;
		bool ok_;

	};
}