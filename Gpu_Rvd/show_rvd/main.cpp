/*
main.cpp
the entrence of the app
*/

#include <graphics\geexapp.h>

#include <iostream>
#include <fstream>
#include <ctime>

namespace Geex{
	class GPURVDApp : public GeexApp{
	public:
		GPURVDApp(int argc, char** argv) : GeexApp(argc, argv) {
			hdr_ = false;
			boundary_filename_ = get_file_arg("obj");
			if (boundary_filename_.length() <= 0){
				std::cerr << "invalid boundary filename" << std::endl;
				exit(0);
			}
			
			points_filename_ = get_file_arg("pts");

			nb_points_ = 100;
			get_arg("nb_pts", nb_points_);
			nb_iters_ = 30;
			get_arg("nb_iter", nb_iters_);
		}

		~GPURVDApp() {
			std::cout << "desconstruct cvt app. " << std::endl;
		}

	private:
		std::string boundary_filename_;
		std::string points_filename_; 
		int nb_points_;
		int nb_iters_;
	};
}

int main(int argc, char** argv){
	srand((unsigned int)time(0));

	Geex::GPURVDApp app(argc, argv);

	return 0;
}