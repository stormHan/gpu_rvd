/*
 * /brief Implementation of computing Restricted Voronoi Diagram.
 */

#include <cuda\cuda_rvd.h>
#include "device_atomic_functions.h"

namespace Gpu_Rvd{

	CudaRestrictedVoronoiDiagram::CudaRestrictedVoronoiDiagram(Mesh* m, Points* p, int iter, int k, int fk){
		vertex_ = m->v_ptr();
		vertex_nb_ = m->get_vertex_nb();
		points_ = p->v_ptr();
		points_nb_ = p->get_vertex_nb();
		facets_ = m->f_ptr();
		facet_nb_ = m->get_facet_nb();
		
		k_ = k;
		fk_ = fk;
		points_nn_ = (index_t*)malloc(sizeof(index_t) * k_ * points_nb_);
		facets_nn_ = (index_t*)malloc(sizeof(index_t) * facet_nb_ * fk_);
		dimension_ = m->dimension();

		dev_vertex_ = nil;
		dev_points_ = nil;
		dev_facets_ = nil;
		dev_points_nn_ = nil;
		dev_facets_nn_ = nil;
		dev_ret_ = nil;
		dev_seeds_info_ = nil;
		dev_seeds_poly_nb = nil;

		mesh_ = m;
		x_ = p;

		NN_ = NearestNeighborSearch::create(dimension_);
		//knn_ = new CudaKNearestNeighbor(*p, *m, k_);
		iter_nb_ = iter;

		is_store_ = true;
		store_filename_counter_ = 0;

		facets_center_ = (double*)malloc(sizeof(double) * dimension_ * facet_nb_);
		index_t f1, f2, f3;
		for (index_t t = 0; t < facet_nb_; ++t){
			f1 = facets_[t * dimension_ + 0];
			f2 = facets_[t * dimension_ + 1];
			f3 = facets_[t * dimension_ + 2];
			Math::compute_center(&vertex_[f1 * 3], &vertex_[f2 * 3], &vertex_[f3 * 3], dimension_, &facets_center_[t * dimension_]);
		}
		
	}

	//CudaRestrictedVoronoiDiagram::CudaRestrictedVoronoiDiagram(Mesh m, Points p, index_t k, const index_t* points_nn, const index_t* facets_nn) :
	//	vertex_(m.v_ptr()),
	//	vertex_nb_(m.get_vertex_nb()),
	//	points_(p.v_ptr()),
	//	points_nb_(p.get_vertex_nb()),
	//	facets_(m.f_ptr()),
	//	facet_nb_(m.get_facet_nb()),
	//	k_(k),
	//	points_nn_(points_nn),
	//	facets_nn_(facets_nn),
	//	dimension_(p.dimension()),
	//	dev_vertex_(nil),
	//	dev_points_(nil),
	//	dev_facets_(nil),
	//	dev_points_nn_(nil),
	//	dev_facets_nn_(nil),
	//	dev_ret_(nil),
	//	host_ret_(nil),
	//	dev_seeds_info_(nil),
	//	dev_seeds_poly_nb(nil)
	//{
	//}

	CudaRestrictedVoronoiDiagram::~CudaRestrictedVoronoiDiagram()
	{
		if (points_nn_ != nil){
			free(points_nn_);
			points_nn_ = nil;
		}
		if (facets_nn_ != nil){
			free(facets_nn_);
			facets_nn_ = nil;
		}

		if (facets_center_ != nil){
			free(facets_center_);
			facets_center_ = nil;
		}
		//delete knn_;
	}

	/*
	* \brief Atomic operation add.
	*
	*/
	__device__
		double MyAtomicAdd(double* address, double val){
		unsigned long long int* address_as_ull = (unsigned long long int*)address;

		unsigned long long int old = *address_as_ull, assumed;

		do{
			assumed = old;
			old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
		} while (assumed != old);

		return __longlong_as_double(old);
	}

	/*
	 * \breif Manipulates the computed RVD data.
	 */
	__device__
		void action(
		const CudaPolygon polygon, index_t current_seed, double* retdata
	){
		double weight;
		double3 position;

		index_t _v1 = 0;
		index_t _v2, _v3;

		double3 pos1, pos2, pos3;
		double d1, d2, d3;
		int triangle_nb = polygon.vertex_nb - 2;
		if (triangle_nb <= 0) return;
		
		double total_weight = 0.0;
		double3 centriodTimesWeight = { 0.0, 0.0, 0.0 };

		double current_weight = 0.0;
		double3 current_posTimesWeight = { 0.0, 0.0, 0.0 };

		//atomicAdd(&g_seeds_polygon_nb[current_seed], 1);
		for (index_t i = 1; i < polygon.vertex_nb - 1; ++i)
		{
			_v2 = i; _v3 = i + 1;

			pos1 = { polygon.vertex[_v1].x, polygon.vertex[_v1].y, polygon.vertex[_v1].z };
			d1 = polygon.vertex[_v1].w;

			pos2 = { polygon.vertex[_v2].x, polygon.vertex[_v2].y, polygon.vertex[_v2].z };
			d2 = polygon.vertex[_v2].w;

			pos3 = { polygon.vertex[_v3].x, polygon.vertex[_v3].y, polygon.vertex[_v3].z };
			d3 = polygon.vertex[_v3].w;

			computeTriangleCentriod(pos1, pos2, pos3, d1, d2, d3, centriodTimesWeight, total_weight);
			
			current_weight += total_weight;
			current_posTimesWeight.x += centriodTimesWeight.x;
			current_posTimesWeight.y += centriodTimesWeight.y;
			current_posTimesWeight.z += centriodTimesWeight.z;

			total_weight = 0.0;
			centriodTimesWeight = { 0.0, 0.0, 0.0 };
		}
		MyAtomicAdd(&retdata[current_seed * 4 + 0], current_posTimesWeight.x);
		MyAtomicAdd(&retdata[current_seed * 4 + 1], current_posTimesWeight.y);
		MyAtomicAdd(&retdata[current_seed * 4 + 2], current_posTimesWeight.z);
		MyAtomicAdd(&retdata[current_seed * 4 + 3], current_weight);
		
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
			index_t j = points_nn[i * k + t];

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
	
	__device__
		void store_info(int facet, int seed, CudaPolygon p, double* address){
		address[0] = facet;
		address[1] = seed;
		address[2] = p.vertex_nb;
		address[3] = 0;
		for (index_t t = 0; t < p.vertex_nb; ++t){
			address[4 + t * 4 + 0] = p.vertex[t].x;
			address[4 + t * 4 + 1] = p.vertex[t].y;
			address[4 + t * 4 + 2] = p.vertex[t].z;
			address[4 + t * 4 + 3] = p.vertex[t].w;
		}
	}
	__global__
		void kernel(
		double*			vertex, index_t			vertex_nb,
		double*			points, index_t			points_nb,
		index_t*		facets, index_t			facets_nb,
		index_t*		points_nn, index_t			k_p,
		index_t*		facets_nn, index_t			k_f,
		index_t			dim, double*			retdata
		){
		index_t t = blockIdx.x * blockDim.x + threadIdx.x;
		if (t >= facets_nb * k_f) return;
		
		index_t tid = index_t(t / k_f);
		index_t pid = t % k_f;

	/*	int cur = tid;
		while (cur < points_nb * 4){
			g_seeds_information[cur] = 0;
			if (cur < points_nb){
				g_seeds_polygon_nb[cur] = 0;
			}
			cur += facets_nb;
		}*/
		//if (tid >= facets_nb) return;

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

		
		
		CudaPolygon current_store = current_polygon;
		//doesn't have the stack?
		index_t to_visit[CUDA_Stack_size];
		index_t to_visit_pos = 0;

		__shared__ index_t has_visited[64][CUDA_Stack_size];
		__shared__ index_t has_visited_nb[64];
		
		bool has_visited_flag = false;
		index_t facetidx_in_block = (index_t)threadIdx.x / k_f;
		int cur = pid;
		while (cur < CUDA_Stack_size){
			has_visited[facetidx_in_block][cur] = -1;
			cur += k_f;
		}
		//debug
		/*__shared__ index_t valid_visited_nb[64];
		valid_visited_nb[facetidx_in_block] = 0;
		__shared__ index_t valid_visited[64][CUDA_Stack_size];
		cur = pid;
		while (cur < CUDA_Stack_size){
			valid_visited[facetidx_in_block][cur] = -1;
			cur += k_f;
		}
		bool valid_visit_flag = false;*/
		//end debug
		has_visited_nb[facetidx_in_block] = 0;
		//load \memory[facets_nn] 1 time.
		to_visit[to_visit_pos++] = facets_nn[pid + tid * k_f];
		//has_visited[has_visited_nb++] = to_visit[0];
		atomicAdd(&has_visited_nb[facetidx_in_block], 1);
		has_visited[facetidx_in_block][pid] = to_visit[0];
		__syncthreads();
		//----------------Time Consuming 1 ms----------------
		//has_visited[facetidx_in_block][has_visited_nb[facetidx_in_block]++] = to_visit[0];
		index_t counter = 0;
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
			
			if (current_polygon.vertex_nb < 3 /*|| current_polygon.vertex_nb > 6*/) break;
			//atomicAdd(&valid_visited_nb[facetidx_in_block], 1);
			//if (tid == facets_nb - 1 && counter == 3){
				/*int idx = 0;
				double* ret = retdata + pid * 32;
				ret[0] = has_visited_nb[facetidx_in_block];
				for (int i = 0; t < has_visited_nb[facetidx_in_block]; ++i){
					ret[i + 1] = has_visited[facetidx_in_block][i];
				}
				
				for (index_t t = 0; t < ret[0]; ++t){
					ret[16 + t] = current_polygon.vertex[t].neigh_s;
				}*/
				//return;
			//}
			return;
			//now we get the clipped polygon stored in "polygon", do something.
			action(
				current_polygon,
				current_seed,
				retdata
				);
			
			//MyAtomicAdd(&retdata[0], 1);
			//store_info(tid, current_seed, current_polygon, &retdata[tid * 400 + counter * 40]);

			//debug
			/*if (counter == 0){
				clock_t start_clock = clock();
				clock_t clock_offset = 0;
				clock_t clock_count = 1000 * pid;
				while (clock_offset < clock_count){
					clock_offset = clock() - start_clock;
				}
			}*/
			/*if (tid == 3092 && pid == 0){
				clock_t start_clock = clock64();
				clock_t clock_offset = 0;
				clock_t clock_count = 5;
				while (clock_offset < 0){
					clock_offset = clock64() - start_clock;
					clock_offset++;
				}
				for (index_t t = 0; t < 20; t++){
					retdata[0] = has_visited_nb[facetidx_in_block];
					retdata[1 + t] = has_visited[facetidx_in_block][t];
				}
				for (index_t t = 0; t < current_polygon.vertex_nb; t++){
					retdata[24 + t] = current_polygon.vertex[t].neigh_s;
					
				}
				return;
			}*/
			//end debug
			//Propagate to adjacent seeds
			for (index_t v = 0; v < current_polygon.vertex_nb; ++v)
			{
				CudaVertex ve = current_polygon.vertex[v];
				int ns = ve.neigh_s;
				
				if (ns != -1 && ns >= 0 && ns < points_nb)
				{
					for (index_t ii = 0; ii < has_visited_nb[facetidx_in_block]; ++ii)
					{
						//if the neighbor seed has clipped the polygon
						//the flag should be set "true"
						if (has_visited[facetidx_in_block][ii] == ns)
							has_visited_flag = true;
					}
					//the neighbor seed is new!
					if (!has_visited_flag)
					{
						to_visit[to_visit_pos++] = ns;
						//has_visited[has_visited_nb++] = ns;
						atomicAdd(&has_visited_nb[facetidx_in_block], 1);
						index_t idx = has_visited_nb[facetidx_in_block] - 1;
						/*while (has_visited[facetidx_in_block][idx] != -1){
							idx++;
						}
						has_visited[facetidx_in_block][idx] = ns;*/
						has_visited[facetidx_in_block][idx] = ns;
					}
					has_visited_flag = false;
				}
			}
			current_polygon = current_store;
			/*current_polygon.vertex_nb = 3;

			current_polygon.vertex[0].x = v1.x; current_polygon.vertex[0].y = v1.y; current_polygon.vertex[0].z = v1.z; current_polygon.vertex[0].w = 1.0;
			current_polygon.vertex[1].x = v2.x; current_polygon.vertex[1].y = v2.y; current_polygon.vertex[1].z = v2.z; current_polygon.vertex[1].w = 1.0;
			current_polygon.vertex[2].x = v3.x; current_polygon.vertex[2].y = v3.y; current_polygon.vertex[2].z = v3.z; current_polygon.vertex[2].w = 1.0;*/
			counter++;
			
		}
		//__syncthreads();
	}

	void CudaRestrictedVoronoiDiagram::knn_search(){
		NN_->set_points(points_nb_, points_);
		update_neighbors();


		/*knn_->set_reference(*x_);
		knn_->set_k(1);
		knn_->set_query(*mesh_);
		knn_->search(facets_nn_);
		knn_->set_k(20);
		knn_->set_query(*x_);
		knn_->search(points_nn_);*/

		//result_print("points_nn.txt", points_nn_, k_ * points_nb_, k_);
		//result_print("facets_nn.txt", facets_nn_, fk_ * facet_nb_, fk_);
	}

	void CudaRestrictedVoronoiDiagram::update_neighbors(){
		long t2 = clock();
		parallel_for(
			parallel_for_member_callback(this, &CudaRestrictedVoronoiDiagram::store_neighbors_CB),
			0, points_nb_, 1, true
			);


		parallel_for(
			parallel_for_member_callback(this, &CudaRestrictedVoronoiDiagram::store_f_neighrbors_CB),
			0, facet_nb_, 1, true
			);
		//double* fp = (double*)malloc(sizeof(double) * dimension_);
		//index_t f1, f2, f3;
		//for (index_t t = 0; t < facet_nb_; ++t){
		//	f1 = facets_[t * dimension_ + 0];
		//	f2 = facets_[t * dimension_ + 1];
		//	f3 = facets_[t * dimension_ + 2];
		//	Math::compute_center(&vertex_[f1 * 3], &vertex_[f2 * 3], &vertex_[f3 * 3], dimension_, fp);
		//	//fp = &vertex_[facets_[t * dimension_] * dimension_];
		//	facets_nn_[t] = NN_->get_nearest_neighbor(fp);
		//}

		/*std::cout << "----- NN TIME : " 
			<< (double)(clock() - t2)
			<< "ms -----"
			<<std::endl;*/
	}

	void CudaRestrictedVoronoiDiagram::store_neighbors_CB(index_t v){
		index_t nb = geo_min(k_, points_nb_);

		// Allocated on the stack(more thread-friendly and 
		// no need to deallocate)
		index_t* neighbors = (index_t*)alloca(
			sizeof(index_t) * nb
			);
		double* dist = (double*)alloca(
			sizeof(double) * nb
			);
		NN_->get_nearest_neighbors(nb, v, neighbors, dist);
		for (index_t t = 0; t < k_; ++t){
			points_nn_[v * k_ + t] = neighbors[t];
		}
	}

	void CudaRestrictedVoronoiDiagram::store_f_neighrbors_CB(index_t v){
		index_t nb = geo_min(fk_, points_nb_);

		// Allocated on the stack(more thread-friendly and 
		// no need to deallocate)
		index_t* neighbors = (index_t*)alloca(
			sizeof(index_t) * nb
			);
		double* dist = (double*)alloca(
			sizeof(double) * nb
			);
		NN_->get_nearest_neighbors(nb, facets_center_ + v * dimension_, neighbors, dist);
		for (index_t t = 0; t < fk_; ++t){
			facets_nn_[v * fk_ + t] = neighbors[t];
		}
	}

	void CudaRestrictedVoronoiDiagram::update_points(){
		x_->clear();

		for (int i = 0; i < points_nb_; ++i){
			if (fabs(host_ret_[i * 4 + 3]) >= 1e-12){
				host_ret_[i * 4 + 0] /= host_ret_[i * 4 + 3];
				host_ret_[i * 4 + 1] /= host_ret_[i * 4 + 3];
				host_ret_[i * 4 + 2] /= host_ret_[i * 4 + 3];
			}
			x_->add_vertexd(&host_ret_[i * 4], dimension_);
		}

		if (is_store_){
			std::string name = "C:\\Users\\JWhan\\Desktop\\DATA\\RVD_" + String::to_string(store_filename_counter_) + ".eobj";
			points_save(name, *x_);
			store_filename_counter_++;
		}
		//debug
		/*std::string name;
		if (store_filename_counter_ != 10)
			name = "C:\\Users\\JWhan\\Desktop\\DATA\\RVD_" + String::to_string(store_filename_counter_) + ".eobj";
		else
			name = "C:\\Users\\JWhan\\Desktop\\DATA\\out.eobj";
		points_load_obj(name, *x_);
		store_filename_counter_++*/;
		//end debug

	}

	__host__
	void CudaRestrictedVoronoiDiagram::compute_Rvd(){
		CudaStopWatcher watcher("compute_rvd_global");
		watcher.start();

		allocate_and_copy(GLOBAL_MEMORY);

		for (index_t t = 0; t < iter_nb_; ++t){
			knn_search(); 
			{
				CudaStopWatcher iter_watcher("iteration");
				iter_watcher.start();				
				cudaMemcpy(dev_points_, points_, DOUBLE_SIZE * points_nb_ * dimension_, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_points_nn_, points_nn_, sizeof(index_t) * points_nb_ * k_, cudaMemcpyHostToDevice);
				cudaMemcpy(dev_facets_nn_, facets_nn_, sizeof(index_t) * facet_nb_ * fk_, cudaMemcpyHostToDevice);
				
				//might be improved dim3 type.
				//int threads = 256;
				//int blocks = facet_nb_ / threads + ((facet_nb_ % threads) ? 1 : 0);
				//dim3 blocks(512, facet_nb_ / 512 + ((facet_nb_ % 512) ? 1 : 0));
				//dim3 threads(fk_, 1, 1);
				int threads = 256;
				int blocks = (fk_ * facet_nb_) / threads + (((fk_ * facet_nb_) % threads) ? 1 : 0);
				kernel << < blocks / 16, threads >> > (
					dev_vertex_, vertex_nb_,
					dev_points_, points_nb_,
					dev_facets_, facet_nb_,
					dev_points_nn_, k_,
					dev_facets_nn_, fk_,
					dimension_, dev_ret_
					);
				CheckCUDAError("kernel function");
				
				copy_back();
				//debug
				//int c   = host_ret_[0];
				//std::ifstream in("C:\\Users\\JWhan\\Desktop\\DATA\\visited_nb.txt");
				//int* temp = (int*)malloc(sizeof(int) * facet_nb_);
				//for (index_t t = 0; t < facet_nb_; ++t){
				//	in >> temp[t];
				//	//std::cout << temp[t];
				//	if ((index_t)host_ret_[t] != (index_t)temp[t]){
				//		std::cout << "Mismatch in facet " << t << " !"
				//			<< "my number = " << host_ret_[t]
				//			<< "  correct number = "
				//			<< temp[t] << std::endl;
				//	}
				//}
				
				//end debug
				//result_print("retdata.txt", host_ret_, points_nb_ * 4, 4);
				is_store_ = false;
				//update_points();
				iter_watcher.stop();
				iter_watcher.synchronize();
				iter_watcher.print_elaspsed_time(std::cout);
			}
		}
		
		watcher.stop();
		watcher.synchronize();
		watcher.print_elaspsed_time(std::cout);
		std::string name = "C:\\Users\\JWhan\\Desktop\\DATA\\RVD_" + String::to_string(store_filename_counter_) + ".eobj";
		points_save(name, *x_);
		free_memory();
	}

	__host__
	void CudaRestrictedVoronoiDiagram::allocate_and_copy(DeviceMemoryMode mode){
		unsigned int free_memory, total_memory;
		cuMemGetInfo(&free_memory, &total_memory);
		std::cerr << "Avaiable GPU memory : " 
			<< free_memory
			<< " Bytes" 
			<< " (Total memory : "
			<< total_memory
			<< " Bytes)"
			<< std::endl
			<< "Starting cudaMalloc..\n";
		host_ret_ = (double*)malloc(sizeof(double) * points_nb_ * (dimension_ + 1));
		//host_ret_ = (double*)malloc(sizeof(double) * facet_nb_ * 10 * 40);
		cudaMalloc((void**)&dev_seeds_info_, DOUBLE_SIZE * points_nb_ * (dimension_ + 1));
		cudaMemcpyToSymbol(g_seeds_information, &dev_seeds_info_, sizeof(double*), size_t(0), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&dev_seeds_poly_nb, INT_SIZE * points_nb_);
		cudaMemcpyToSymbol(g_seeds_polygon_nb, &dev_seeds_poly_nb, sizeof(int*), size_t(0), cudaMemcpyHostToDevice);

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
			cudaMalloc((void**)&dev_facets_nn_, sizeof(index_t) * facet_nb_ * fk_);

			//Output result.
			//cudaMalloc((void**)&dev_ret_, sizeof(double) *  facet_nb_ * 10 * 40);
			cudaMalloc((void**)&dev_ret_, sizeof(double) * points_nb_ * 4);
			CheckCUDAError("Allocating device memory");

			//Copy
			cudaMemcpy(dev_vertex_, vertex_, DOUBLE_SIZE * vertex_nb_ * dimension_, cudaMemcpyHostToDevice);
			//cudaMemcpy(dev_points_, points_, DOUBLE_SIZE * points_nb_ * dimension_, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_facets_, facets_, sizeof(index_t) * facet_nb_ * dimension_, cudaMemcpyHostToDevice);
			//cudaMemcpy(dev_points_nn_, points_nn_, sizeof(index_t) * points_nb_ * k_, cudaMemcpyHostToDevice);
			//cudaMemcpy(dev_facets_nn_, facets_nn_, sizeof(index_t) * facet_nb_ * 1, cudaMemcpyHostToDevice);
			cuMemGetInfo(&free_memory, &total_memory);
			std::cerr << "Left GPU memory : "
				<< free_memory
				<< " Bytes"
				<< " (Total memory : "
				<< total_memory
				<< " Bytes)"
				<< std::endl;
			CheckCUDAError("Copying data from host to device");
		}
			break;
		case CONSTANT_MEMORY:
		{
			cudaMalloc((void**)&dev_facets_, sizeof(index_t) * facet_nb_ * dimension_);
			cudaMalloc((void**)&dev_facets_nn_, sizeof(index_t) * facet_nb_ * 1);

			//Output result.
			cudaMalloc((void**)&dev_ret_, sizeof(double) * points_nb_ * 4);
			CheckCUDAError("Allocating device memory");

			cudaMemcpyToSymbol(c_vertex, vertex_, DOUBLE_SIZE * vertex_nb_ * dimension_);
			cudaMemcpyToSymbol(c_points, points_, DOUBLE_SIZE * points_nb_ * dimension_);
			cudaMemcpyToSymbol(c_points_nn, points_nn_, INT_SIZE * points_nb_ * k_);

			cudaMemcpy(dev_facets_nn_, facets_nn_, sizeof(index_t) * facet_nb_ * 1, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_facets_, facets_, sizeof(index_t) * facet_nb_ * dimension_, cudaMemcpyHostToDevice);
			CheckCUDAError("Copying data from host to device");
		}
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
		cudaFree(dev_ret_);
		cudaFree(dev_seeds_info_);
		cudaFree(dev_seeds_poly_nb);
		if (host_ret_ != nil){
			free(host_ret_);
			host_ret_ = nil;
		}
	}

	__host__
	void CudaRestrictedVoronoiDiagram::copy_back(){
		//cudaMemcpy(host_ret_, dev_ret_, sizeof(double) * facet_nb_ * 10 * 40, cudaMemcpyDeviceToHost);
		cudaMemcpy(host_ret_, dev_ret_, sizeof(double) * points_nb_ * 4, cudaMemcpyDeviceToHost);
		CheckCUDAError("copy back");
	}

	__host__
		void CudaRestrictedVoronoiDiagram::print_return_data(std::string filename) const{
		for (int i = 0; i < points_nb_; ++i)
		{
			if (fabs(host_ret_[i * 4 + 3]) >= 1e-12){
				host_ret_[i * 4 + 0] /= host_ret_[i * 4 + 3];
				host_ret_[i * 4 + 1] /= host_ret_[i * 4 + 3];
				host_ret_[i * 4 + 2] /= host_ret_[i * 4 + 3];
			}
		}
		index_t line_num = 4;
		std::ofstream f;
		f.open(filename);
		for (index_t t = 0; t < facet_nb_ * 100; ++t){
		//for (index_t t = 0; t < points_nb_; ++t){
			f << std::setprecision(18);
			f << "point " << t << " ";
			f << host_ret_[t * 4 + 0] << " "
				<< host_ret_[t * 4 + 1] << " "
				<< host_ret_[t * 4 + 2] << " "
				<< host_ret_[t * 4 + 3] << " " << std::endl;
		}
		f.close();
	}
}