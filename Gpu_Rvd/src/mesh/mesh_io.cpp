
#include <mesh\mesh_io.h>

namespace Gpu_Rvd{

	bool mesh_load_obj(const std::string filename, Mesh& M){
		coords_index_t dim = M.dimension();
		std::vector<double> P(dim);

		LineInput in(filename);
		if (!in.OK()){
			return false;
		}

		std::vector<index_t> facet_vertices;
		bool first_facet_attribute = true;
		bool read_facet_regions = false;
		while (!in.eof() && in.get_line()){
			in.get_fields(); //must be called after get_line()
			if (in.nb_fields() >= 1){
				if (in.field_matches(0, "v")){
					for (coords_index_t c = 0; c < dim; ++c){
						if (index_t(c + 1) < in.nb_fields()){
							P[c] = in.field_as_double(index_t(c + 1));
						}
						else{
							P[c] = 0.0;
						}
					}
					M.add_vertexd(P);
				}
				else if (in.field_matches(0, "f")){
					if (in.nb_fields() < 3){
						fprintf(stderr, "facet only has %d corners(at least 3 required)", in.nb_fields());
						return false;
					}
					facet_vertices.resize(0);
					
					for (index_t i = 1; i < in.nb_fields(); ++i){
						for (char* ptr = in.field(i); *ptr != '\0'; ++ptr){
							if (*ptr == '/'){
								*ptr = '\0';
								break;
							}
						}
						index_t vertex_index = in.field_as_uint(i);
						if (
							(vertex_index < 1) ||
							(vertex_index > M.get_vertex_nb())
							){
							fprintf(stderr, "facet corner %d references an invalid vertex index", in.nb_fields());
							return false;
						}
						facet_vertices.push_back(vertex_index - 1);
					}
					M.add_facet(facet_vertices);
				}
			}
		}
		return true;
	}

	bool points_load_obj(const std::string filename, Points& points){
		coords_index_t dim = points.dimension();
		std::vector<double> P(dim);

		LineInput in(filename);
		if (!in.OK()){
			return false;
		}
		bool first_facet_attribute = true;
		bool read_facet_regions = false;
		while (!in.eof() && in.get_line()){
			in.get_fields(); //must be called after get_line()
			if (in.nb_fields() >= 1){
				if (in.field_matches(0, "v")){
					for (coords_index_t c = 0; c < dim; ++c){
						if (index_t(c + 1) < in.nb_fields()){
							P[c] = in.field_as_double(index_t(c + 1));
						}
						else{
							P[c] = 0.0;
						}
					}
					points.add_vertexd(P);
				}
			}
		}
		return true;
	}

}