/*
 * /brief Implementation of computing Restricted Voronoi Diagram.
 */

#include <cuda\cuda_rvd.h>

namespace Gpu_Rvd{

	CudaRestrictedVoronoiDiagram::CudaRestrictedVoronoiDiagram(Mesh m, Points p) : 
		vertex_(m.v_ptr()),
		vertex_nb_(m.get_vertex_nb()),
		points_(p.v_ptr()),
		points_nb_(p.get_vertex_nb()),
		facets_(m.f_ptr()),
		facet_nb_(m.get_facet_nb()),
		k_(p.get_k()),
		points_nn_(p.get_indices()),
		facets_nn_(m.get_indices()),
		dimension_(p.dimension()),
		dev_vertex_(nil),
		dev_points_(nil),
		dev_facets_(nil),
		dev_points_nn_(nil),
		dev_facets_nn_(nil),
		dev_ret(nil),
		host_ret(nil)
	{
	}

	CudaRestrictedVoronoiDiagram::CudaRestrictedVoronoiDiagram(Mesh m, Points p, index_t k, const index_t* points_nn, const index_t* facets_nn) :
		vertex_(m.v_ptr()),
		vertex_nb_(m.get_vertex_nb()),
		points_(p.v_ptr()),
		points_nb_(p.get_vertex_nb()),
		facets_(m.f_ptr()),
		facet_nb_(m.get_facet_nb()),
		k_(k),
		points_nn_(points_nn),
		facets_nn_(facets_nn),
		dimension_(p.dimension()),
		dev_vertex_(nil),
		dev_points_(nil),
		dev_facets_(nil),
		dev_points_nn_(nil),
		dev_facets_nn_(nil),
		dev_ret(nil),
		host_ret(nil)
	{
	}

	CudaRestrictedVoronoiDiagram::CudaRestrictedVoronoiDiagram(
		const double* vertex, index_t vertex_nb,
		const double* points, index_t points_nb,
		const index_t* facets, index_t facets_nb,
		index_t* points_nn, index_t k_p,
		index_t* facets_nn, index_t dim
		) :
		vertex_(vertex),
		vertex_nb_(vertex_nb),
		points_(points),
		points_nb_(points_nb),
		facets_(facets),
		facet_nb_(facets_nb),
		k_(k_p),
		points_nn_(points_nn),
		facets_nn_(facets_nn),
		dimension_(dim),
		dev_vertex_(nil),
		dev_points_(nil),
		dev_facets_(nil),
		dev_points_nn_(nil),
		dev_facets_nn_(nil),
		dev_ret(nil),
		host_ret(nil)
	{
	}

	CudaRestrictedVoronoiDiagram::~CudaRestrictedVoronoiDiagram()
	{
	}

	/*
	 * \breif Manipulates the computed RVD data.
	 */
	__device__
	void action(){

	}

	/*
	 * \brief Clips the Polygon by the middle plane defined by point i and j.
	 */
	__device__
		void clip_by_plane(
		CudaPolygon& ping,
		CudaPolygon& pong,
		double3 position_i,
		double3 position_j,
		index_t j
		){

		//reset the pong
		pong.vertex_nb = 0;

		if (ping.vertex_nb == 0)
			return;

		// Compute d = n . (2m), where n is the
		// normal vector of the bisector [i, j]
		// and m the middle point of the bisector.
		double d = 0.0;
		d = dot(add(position_i, position_j), sub(position_i, position_j));

		//The predecessor of the first vertex is the last vertex
		index_t prev_k = ping.vertex_nb - 1;

		//get the position data
		CudaVertex* prev_vk = &ping.vertex[prev_k];

		double3 prev_vertex_position = { prev_vk->x, prev_vk->y, prev_vk->z };

		//then we compute prev_vertex_position "cross" n 
		//prev_l = prev_vertex_position . n
		double prev_l = dot(prev_vertex_position, sub(position_i, position_j));

		int prev_status = sgn(2.0 * prev_l - d);

		//traverse the Vertex in this Polygon
		for (index_t k = 0; k < ping.vertex_nb; ++k){

			CudaVertex* vk = &ping.vertex[k];
			double3 vertex_position = { vk->x, vk->y, vk->z };

			double l = dot(vertex_position, sub(position_i, position_j));
			int status = sgn(2.0 * l - d);

			//If status of edge extremities differ,
			//then there is an intersection.
			if (status != prev_status && (prev_status) != 0){
				// create the intersection and update the Polyon
				CudaVertex I;

				//compute the position and weight
				double denom = 2.0 * (prev_l - l);
				double lambda1, lambda2;

				// Shit happens!
				if (m_fabs(denom) < 1e-20)
				{
					lambda1 = 0.5;
					lambda2 = 0.5;
				}
				else
				{
					lambda1 = (d - 2.0 * l) / denom;
					// Note: lambda2 is also given
					// by (2.0*l2-d)/denom
					// (but 1.0 - lambda1 is a bit
					//  faster to compute...)
					lambda2 = 1.0 - lambda1;
				}

				//Set the Position of Vertex
				I.x = lambda1 * prev_vertex_position.x + lambda2 * vertex_position.x;
				I.y = lambda1 * prev_vertex_position.y + lambda2 * vertex_position.y;
				I.z = lambda1 * prev_vertex_position.z + lambda2 * vertex_position.z;

				//Set the Weight of Vertex
				I.w = (lambda1 * prev_vk->w + lambda2 * vk->w);

				if (status > 0)
				{
					I.neigh_s = (j);
				}
				else {
					I.neigh_s = (vk->neigh_s);
				}

				//add I to pong
				pong.vertex[pong.vertex_nb] = I;
				pong.vertex_nb++;
			}
			if (status > 0)
			{
				//add vertex to pong
				pong.vertex[pong.vertex_nb] = *vk;
				pong.vertex_nb++;
			}

			prev_vk = vk;
			prev_vertex_position = vertex_position;
			prev_status = status;
			prev_l = l;
			prev_k = k;
		}
	}

	/*
	 * \brief Swaps the content of ping and pong.
	 * stores the result in ping.
	 */
	__device__
	void swap_polygon(CudaPolygon& ping, CudaPolygon& pong){
		CudaPolygon t = ping;
		ping = pong;
		pong = t;
	}


	/*
	 * \brief Intersects a polygon with a points.
	 */
	__device__
	void intersection_clip_facet_SR(
		CudaPolygon& current_polygon,
		index_t i,
		const double* points,
		index_t points_nb,
		index_t* points_nn,
		index_t k
	){
		CudaPolygon polygon_buffer;
		
		//load /memory[points] 3 times.
		double3 pi = {
			points[i * 3 + 0],
			points[i * 3 + 1],
			points[i * 3 + 2]
		};

		for (index_t t = 0; t < k; ++t){

			//load /memory[points_nn] k times.
			index_t j = points[i * k + t];

			if (i != j){
				//load /memroy[points] k * 3 times.
				double3 pj = {
					points[j * 3 + 0],
					points[j * 3 + 1],
					points[j * 3 + 2]
				};

				double dij = distance2(pi, pj);
				double R2 = 0.0;

				for (index_t tt = 0; tt < current_polygon.vertex_nb; ++tt){
					double3 pk = { current_polygon.vertex[tt].x, current_polygon.vertex[tt].y, current_polygon.vertex[tt].z };
					double dik = distance2(pi, pk);
					R2 = max(R2, dik);
				}
				if (dij > 4.1 * R2){
					return;
				}
				clip_by_plane(current_polygon, polygon_buffer, pi, pj, j);
				swap_polygon(current_polygon, polygon_buffer);
			}
		}
	}
	

	__global__
	void kernel(
		double*			vertex,		index_t			vertex_nb,
		double*			points,		index_t			points_nb,
		index_t*		facets,		index_t			facets_nb,
		index_t*		points_nn,	index_t			k_p,
		index_t*		facets_nn,	index_t			dim,
		double*			retdata
		){
		index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= facets_nb) return;

		//load \memory[facet] 3 times.
		int3 facet_index = {
			facets[tid * dim + 0],
			facets[tid * dim + 1],
			facets[tid * dim + 2]
		};
		
		//load \memory[vertex] 9 times.
		double3 v1 = {
			vertex[facet_index.x * dim + 0],
			vertex[facet_index.x * dim + 1],
			vertex[facet_index.x * dim + 2]
		};
		double3 v2 = {
			vertex[facet_index.y * dim + 0],
			vertex[facet_index.y * dim + 1],
			vertex[facet_index.y * dim + 2]
		};
		double3 v3 = {
			vertex[facet_index.z * dim + 0],
			vertex[facet_index.z * dim + 1],
			vertex[facet_index.z * dim + 2]
		};

		CudaPolygon current_polygon;
		current_polygon.vertex_nb = 3;

		current_polygon.vertex[0].x = v1.x; current_polygon.vertex[0].y = v1.y; current_polygon.vertex[0].z = v1.z; current_polygon.vertex[0].w = 1.0;
		current_polygon.vertex[1].x = v2.x; current_polygon.vertex[1].y = v2.y; current_polygon.vertex[1].z = v2.z; current_polygon.vertex[1].w = 1.0;
		current_polygon.vertex[2].x = v3.x; current_polygon.vertex[2].y = v3.y; current_polygon.vertex[2].z = v3.z; current_polygon.vertex[2].w = 1.0;

		//doesn't have the stack?
		index_t to_visit[CUDA_Stack_size];
		index_t to_visit_pos = 0;

		index_t has_visited[CUDA_Stack_size];
		index_t has_visited_nb = 0;
		bool has_visited_flag = false;

		//load \memory[facets_nn] 1 time.
		to_visit[to_visit_pos++] = facets_nn[tid];
		has_visited[has_visited_nb++] = to_visit[0];

		while (to_visit_pos){
			index_t current_seed = to_visit[to_visit_pos - 1];
			to_visit_pos--;

			intersection_clip_facet_SR(
				current_polygon,
				current_seed,
				points,
				points_nb,
				points_nn,
				k_p
				);

			//now we get the clipped polygon stored in "polygon", do something.
			action(
				);

			//Propagate to adjacent seeds
			for (index_t v = 0; v < current_polygon.vertex_nb; ++v)
			{
				CudaVertex ve = current_polygon.vertex[v];
				int ns = ve.neigh_s;
				if (ns != -1)
				{
					for (index_t ii = 0; ii < has_visited_nb; ++ii)
					{
						//if the neighbor seed has clipped the polygon
						//the flag should be set "true"
						if (has_visited[ii] == ns)
							has_visited_flag = true;
					}
					//the neighbor seed is new!
					if (!has_visited_flag)
					{
						to_visit[to_visit_pos++] = ns;
						has_visited[has_visited_nb++] = ns;
					}
					has_visited_flag = false;
				}
			}
		}
		if (tid == 0){
			retdata[0] = current_polygon.vertex_nb;
			
		}
	}

	__host__
	void CudaRestrictedVoronoiDiagram::compute_Rvd(){
		CudaStopWatcher watcher("compute_rvd");
		watcher.start();

		allocate_and_copy(GLOBAL_MEMORY);
		//might be improved dim3 type.
		int threads = 512;
		int blocks = facet_nb_ / threads + ((facet_nb_ % threads) ? 1 : 0);
		kernel << < threads, blocks >> > (
			dev_vertex_, vertex_nb_,
			dev_points_, points_nb_,
			dev_facets_, facet_nb_,
			dev_points_nn_, k_,
			dev_facets_nn_, dimension_,
			dev_ret
			);
		CheckCUDAError("kernel function");
		copy_back();
		watcher.stop();
		watcher.synchronize();
		watcher.print_elaspsed_time(std::cout);
		
		std::string out_file("..//out//retdata.txt");
		print_return_data(out_file);
		free_memory();
	}

	__host__
	void CudaRestrictedVoronoiDiagram::allocate_and_copy(DeviceMemoryMode mode){

		host_ret = (double*)malloc(sizeof(double) * points_nb_ * 4);
		switch (mode)
		{
		case GLOBAL_MEMORY:
		{
			//Allocate
			//Input data.
			cudaMalloc((void**)&dev_vertex_, DOUBLE_SIZE * vertex_nb_ * dimension_);
			cudaMalloc((void**)&dev_points_, DOUBLE_SIZE * points_nb_ * dimension_);
			cudaMalloc((void**)&dev_facets_, sizeof(index_t) * facet_nb_ * dimension_);
			cudaMalloc((void**)&dev_points_nn_, sizeof(index_t) * points_nb_ * k_);
			cudaMalloc((void**)&dev_facets_nn_, sizeof(index_t) * facet_nb_ * 1);

			//Output result.
			cudaMalloc((void**)&dev_ret, sizeof(double) * points_nb_ * 4);
			CheckCUDAError("Allocating device memory");

			//Copy
			cudaMemcpy(dev_vertex_, vertex_, DOUBLE_SIZE * vertex_nb_ * dimension_, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_points_, points_, DOUBLE_SIZE * points_nb_ * dimension_, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_facets_, facets_, sizeof(index_t) * facet_nb_ * dimension_, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_points_nn_, points_nn_, sizeof(index_t) * points_nb_ * k_, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_facets_nn_, facets_nn_, sizeof(index_t) * facet_nb_ * 1, cudaMemcpyHostToDevice);

			CheckCUDAError("Copying data from host to device");
		}
			break;
		case CONSTANT_MEMORY:
			break;
		case TEXTURE_MEMORY:
			break;
		default:
			break;
		}
	}

	__host__
	void CudaRestrictedVoronoiDiagram::free_memory(){
		cudaFree(dev_vertex_);
		cudaFree(dev_points_);
		cudaFree(dev_facets_);
		cudaFree(dev_points_nn_);
		cudaFree(dev_facets_nn_);
		cudaFree(dev_ret);

		if (host_ret != nil){
			free(host_ret);
			host_ret = nil;
		}
	}

	__host__
	void CudaRestrictedVoronoiDiagram::copy_back(){
		cudaMemcpy(host_ret, dev_ret, sizeof(double) * points_nb_ * 4, cudaMemcpyDeviceToHost);
		CheckCUDAError("copy back");
	}

	__host__
		void CudaRestrictedVoronoiDiagram::print_return_data(std::string filename) const{
		index_t line_num = 4;
		std::ofstream f;
		f.open(filename);
		for (index_t t = 0; t < points_nb_; ++t){
			f << std::setprecision(18);
			f << "point " << t << " ";
			f << host_ret[t] << " "
				<< host_ret[t + 1] << " "
				<< host_ret[t + 2] << " "
				<< host_ret[t + 3] << " " << std::endl;
		}
		f.close();
	}
}