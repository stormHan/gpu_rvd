/************************************

	implematation of Input the Mesh

************************************/

#ifndef MESH_MESH_IO_H
#define MESH_MESH_IO_H

#include <basic\common.h>
#

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
	bool mesh_load(const std::string _filepath, Mesh& _M, bool _meshpoints = true, FileType _filetype = OBJfile);

	/*
	save the mesh.
	*/

}

#endif /* MESH_MESH_IO_H */