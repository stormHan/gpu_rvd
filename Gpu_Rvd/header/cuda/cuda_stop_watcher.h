/*
 * Cuda Stop watcher implementation
 *
 */

#ifndef CUDA_STOP_WATCHER_H
#define CUDA_STOP_WATCHER_H

#include <basic\common.h>
#include <cuda\cuda_common.h>
namespace Gpu_Rvd{
	
	/*
	* \brief Cuda stopwatcher manipulation utilities.
	*/
	class CudaStopWatcher{
	public:
		/*
		* \brief Creates the cuda event to record the time.
		*/
		CudaStopWatcher(std::string name) :
			name_(name)
		{
			cudaEventCreate(&start_);
			cudaEventCreate(&stop_);
		};

		/*
		* \brief Destory the cuda event.
		*/
		~CudaStopWatcher(){
			cudaEventDestroy(start_);
			cudaEventDestroy(stop_);
		}

		/*
		* \brief Cuda stopwatcher starts to record the time.
		*/
		void start(){
			cudaEventRecord(start_, 0);
		}

		/*
		* \brief Cuda stopwatcher stops.
		*/
		void stop(){
			cudaEventRecord(stop_, 0);
		}

		/*
		* \brief Cuda stopwatcher shows the current time.
		*/
		void print_elaspsed_time(std::ostream& os){
			cudaEventElapsedTime(&time_, start_, stop_);

			os << "---- Times (ms) ----\n"
				<< "---- Task :" << name_ << "----"
				<< "\n  Elapsed time: " << time_
				<< std::endl;
		}

		/*
		 * \brief Synchronizes to wait for an event to complete.
		 */
		void synchronize(){
			cudaEventSynchronize(start_);
			cudaEventSynchronize(stop_);
		}
	private:
		cudaEvent_t start_, stop_;;
		float time_;
		std::string name_;
	};
}
#endif /* CUDA_STOP_WATCHER_H */
