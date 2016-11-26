/*
	implementation of the line reader
*/

#include <basic\line_stream.h>

namespace Gpu_Rvd{

	LineInput::LineInput(const std::string& filename):
		file_name_(filename),
		line_num_(0)
	{
		F_ = fopen(filename.c_str(), "r");
		ok_ = (F_ != nil);
		line_[0] = '\0';	//end char
	}

	LineInput::~LineInput(){
		if (F_ != nil){
			fclose(F_);
			F_ = nil;
		}
	}

	bool LineInput::get_line(){
		if (F_ == nil){
			return false;
		}

		line_[0] = '\0';

		//Skip the empty lines
		while (!isprint(line_[0])){
			++line_num_;
			if (fgets(line_, MAX_LINE_LEN, F_) == nil){
				return false;
			}
		}

		bool check_multiline = true;
		int64 total_length = MAX_LINE_LEN;
		char* ptr = line_;
		while (check_multiline){
			size_t L = strlen(ptr);
			total_length -= int64(L);
			ptr = ptr + L - 2;
			if (*ptr == '\\' && total_length > 0) {
				*ptr = ' ';
				ptr++;
				if (fgets(ptr, int(total_length), F_) == nil) {
					return false;
				}
				++line_num_;
			}
			else {
				check_multiline = false;
			}
		}
		if (total_length < 0){
			fprintf(stderr, "MultiLine longer than MAX_LEN!");
			return false;
		}
		return true;
	}

	void LineInput::get_fields(const char* separators){
		field_.resize(0);
		char* context = nil;
		char* tok = strtok_s(line_, separators, &context);
		while (tok != nil){
			field_.push_back(tok);
			tok = strtok_s(nil, separators, &context);
		}
	}
}