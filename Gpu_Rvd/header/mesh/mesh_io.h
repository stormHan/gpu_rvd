/************************************

	implematation of Input the Mesh

************************************/

#ifndef MESH_MESH_IO_H
#define MESH_MESH_IO_H

#include <basic\common.h>
#include <basic\line_stream.h>
#include <mesh\mesh.h>
#include <iomanip>

namespace Gpu_Rvd{

	enum FileType
	{
		OBJfile = 0,
		PTXfile = 1
	};
	/*
	load the mesh attributes from the file

	meshpoints : true	-- load a mesh
	flase	-- load points, just vertices
	*/
	bool mesh_load_obj(const std::string filename, Mesh& M);


	bool points_load_obj(const std::string filename, Points& M);
	/*
	save the mesh.
	*/

	bool points_save(const std::string& filename, Points& p);

	/**
	 * \brief save points to .xyz file
	 */
	bool points_save_xyz(const std::string& filename, Points& p,const std::vector<int>& sample_facet);
}

#endif /* MESH_MESH_IO_H */