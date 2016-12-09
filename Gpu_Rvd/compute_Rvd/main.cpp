/*
	main.cpp 
	the entrence of the app
*/

#include <basic\common.h>
#include <basic\command_line.h>
#include <basic\stop_watch.h>
#include <mesh\mesh.h>
#include <mesh\mesh_io.h>
#include <cuda\cuda_common.h>
#include <cuda\cuda_rvd.h>

int main(int argc, char** argv){
	using namespace Gpu_Rvd;

	StopWatch S("total task");
	Cmd::Mode mode = Cmd::Host_Device;
	std::vector<std::string> filenames;
	if (!Cmd::parse_argu(argc, argv, filenames, mode)){
		fprintf(stderr, "failed to parse the argument!");
		return 1;
	}

	std::string mesh_filename = filenames[0];
	std::string points_filename = filenames[0];
	std::string output_filename;
	if (filenames.size() >= 2) {
		points_filename = filenames[1];
	}
	output_filename = (filenames.size() == 3) ? filenames[2] : "out.eobj";

	Mesh M_in;
	Points Points_in;

	if (!mesh_load_obj(mesh_filename, M_in)){
		fprintf(stderr, "cannot load the mesh from the %s path", mesh_filename);
		return 1;
	}

	if (!points_load_obj(points_filename, Points_in)){
		fprintf(stderr, "cannot load the points from the %s path", points_filename);
		return 1;
	}
	

#ifdef KNN
	
#else
	// do not use Knn algorigthm, find the Nearest Neighbors by comparing the distance in CPU.
	//Points_in.search_nn(Points_in.v_ptr(), 3, Points_in.get_vertex_nb(), 20);
	
	index_t* points_nn = (index_t*)malloc(sizeof(index_t) * Points_in.get_vertex_nb() * 20);
	index_t* facets_nn = (index_t*)malloc(sizeof(index_t) * M_in.get_facet_nb());

	freopen("..//test//right//bunny_points_nn.txt", "r", stdin);
	for (index_t t = 0; t < Points_in.get_vertex_nb() * 20; ++t){
		scanf("%d ", &points_nn[t]);
	}
	freopen("..//test//right//bunny_facets_nn.txt", "r", stdin);
	for (index_t t = 0; t < M_in.get_facet_nb(); ++t){
		scanf("%d ", &facets_nn[t]);
	}
	freopen("CON", "r", stdin);
	/*freopen("..//test//S2_points_nn", "w", stdout);
	for (index_t t = 0; t < Points_in.get_vertex_nb() * 20; ++t){
		printf("%d ", points_nn[t]);
		if (t % 20 == 19) printf("\n");
	}
	printf("\n");
	for (int i = 0; i < 1000; ++i){
		printf("%d ", i);
	}
	freopen("..//test//S2_facets_nn", "w", stdout);
	for (index_t t = 0; t < M_in.get_facet_nb(); ++t){
		printf("%d %d\n",t, facets_nn[t]);
	}
	printf("\n");
	for (int i = 0; i < 1000; ++i){
		printf("%d ", i);
	}*/
#endif
	CudaRestrictedVoronoiDiagram RVD(
		M_in.v_ptr(),			M_in.get_vertex_nb(),
		Points_in.v_ptr(),		Points_in.get_vertex_nb(),
		M_in.f_ptr(),			M_in.get_facet_nb(),
		points_nn,				20,
		facets_nn,				Points_in.dimension()
		);

	RVD.compute_Rvd();

	S.print_elaspsed_time(std::cout);
	getchar();
	return 0;
}