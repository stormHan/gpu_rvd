
#include <basic\process.h>

namespace {
	using namespace Gpu_Rvd;
}
namespace {

	/**
	* \brief The (thread-local) variable that stores a
	*  pointer to the current thread.
	* \details It cannot be a static member of class
	*  Thread, because Visual C++ does not accept
	*  to export thread local storage variables in
	*  DLLs.
	*/
	GEO_THREAD_LOCAL Thread*  geo_current_thread_ = nil;

}
namespace Gpu_Rvd{

	void Thread::set_current(Thread* thread){
		geo_current_thread_ = thread;
	}

	Thread* Thread::current(){
		return geo_current_thread_;
	}

	Thread::~Thread(){}

	/************************************************************************/

	ThreadManager::~ThreadManager(){}

	void ThreadManager::run_threads(ThreadGroup& threads){
		index_t max_threads = maximum_concurrent_threads();

	}

	/************************************************************************/

	void MonoThreadingThreadManager::enter_critical_section() {
	}

	void MonoThreadingThreadManager::leave_critical_section() {
	}


	MonoThreadingThreadManager::~MonoThreadingThreadManager() {
	}

	void MonoThreadingThreadManager::run_concurrent_threads(
		ThreadGroup& threads, index_t max_threads
		) {
	}

	/************************************************************************/
}