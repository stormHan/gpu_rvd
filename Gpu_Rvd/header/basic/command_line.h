/*
	handle the command argument
	type is as follow:
		compute_Rvd.exe <Mesh_path> <Points_path> <Out_file_path> <mode>
*/

#ifndef BASIC_COMMAND_LINE_H
#define BASIC_COMMAND_LINE_H

#include <basic\common.h>

namespace Gpu_Rvd{
	namespace Cmd{

		enum Mode{
			Host = 0,
			Device = 1,
			Host_Device = 2
		};

		/*
			parse the command line.

		*/
		bool parse_argu(index_t argument_nb, char** argument, std::vector<std::string>& filenames);

	}
}

#endif /* BASIC_COMMAND_LINE_H */