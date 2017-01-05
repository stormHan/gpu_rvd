/*
	main.cpp 
	the entrence of the app
*/

#include <basic\common.h>
#include <basic\command_line.h>
#include <basic\stop_watch.h>
#include <basic\process.h>
#include <mesh\mesh.h>
#include <mesh\mesh_io.h>
#include <cuda\cuda_common.h>
#include <cuda\cuda_rvd.h>
#include <cuda\cuda_knn.h>

#include <ctime>

#define KNN

int main(int argc, char** argv){
	using namespace Gpu_Rvd;
	Process::initialize();

	srand((unsigned int)time(0));
	const index_t iteration = 100;

	StopWatch S("total task");
	std::vector<std::string> filenames;
	if (!Cmd::parse_argu(argc, argv, filenames)){
		fprintf(stderr, "failed to parse the argument!");
		return 1;
	}

	std::string mesh_filename = filenames[0];
	std::string points_filename = filenames[0];
	std::string output_filename;
	if (filenames.size() >= 2) {
		points_filename = filenames[1];
	}
	output_filename = (filenames.size() == 3) ? filenames[2] : "C:\\Users\\JWhan\\Desktop\\DATA\\out.eobj";

	Mesh M_in;
	Points Points_in;

	if (!mesh_load_obj(mesh_filename, M_in)){
		fprintf(stderr, "cannot load the mesh from the %s path", mesh_filename);
		return 1;
	}
	//points_filename = "C:\\Users\\JWhan\\Desktop\\DATA\\out.eobj";
	points_filename = "D:\\Project\\Models\\bunny.obj";

	if (!points_load_obj(points_filename, Points_in)){
		fprintf(stderr, "cannot load the points from the %s path", points_filename);
		return 1;
	}
	//M_in.init_samples(Points_in, 200000);
	
	/*if (!points_save(output_filename, Points_in)){
		std::cerr << "cannot save the points data" << std::endl;
		return 1;
	}*/
	
	//default settings: k = 20, store_mode
	CudaRestrictedVoronoiDiagram rvd(&M_in, &Points_in, iteration);
	rvd.compute_Rvd();


	S.print_elaspsed_time(std::cout);
	Process::terminate();
	return 0;
}