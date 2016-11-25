/*
	
*/

#include <basic\command_line.h>

namespace Gpu_Rvd{
	namespace Cmd{
		bool parse_argu(int argument_nb, char** argument, std::vector<std::string>& filenames, Mode& mode){
			if (argument_nb <= 1){
				fprintf(stderr, "no argument pointed to filenames!");
				return false;
			}
			if (argument_nb > 5){
				fprintf(stderr, "too many argument!");
				return false;
			}
			
			for (index_t i = 1; i < argument_nb; ++i){
				if (i < argument_nb - 1){
					filenames.push_back(argument[i]);
				}
				else{
					std::string t_str = argument[i];
					if (t_str == "CPU") mode = Cmd::Host;
					else if (t_str == "GPU") mode = Cmd::Device;
					else if (t_str == "CPU/GPU") mode = Cmd::Host_Device;
					else filenames.push_back(argument[i]);
				}
			}
			return true;
		}
	}
}