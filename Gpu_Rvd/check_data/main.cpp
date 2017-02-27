/*
 * \brief Checks the data from file
 */

#include <basic\common.h>
#include <mesh\mesh.h>
#include <mesh\mesh_io.h>

using namespace Gpu_Rvd;

int main(){
	std::vector<int> sample_facet;
	Points m_ans, right_ans;
	points_load_xyz("C:\\Users\\JWhan\\Desktop\\DATA\\after.txt", m_ans, sample_facet);
	points_load_xyz("C:\\Users\\JWhan\\Desktop\\DATA\\RVDans.xyz", right_ans, sample_facet);

	//computing the wrong points
	if (m_ans.get_vertex_nb() != right_ans.get_vertex_nb()){
		std::cout << "cannot be compared for two file vertex number is different"
			<< std::endl;
	}
	std::fstream check("C:\\Users\\JWhan\\Desktop\\DATA\\check.txt");
	double cur_d = 0.0;
	double d = 0.0001;
	for (index_t t = 0; t < right_ans.get_vertex_nb(); ++t){
		for (index_t tt = 0; tt < right_ans.dimension(); ++tt){
			cur_d += fabs(m_ans.get_vertexd(t)[tt] - right_ans.get_vertexd(t)[tt]);
		}
		if (cur_d > d){
			std::cout << "wrongly point : " << t << std::endl;
			check << "wrongly point : " << t << std::endl;
		}
		cur_d = 0.0;
	}
	return 0;
}

