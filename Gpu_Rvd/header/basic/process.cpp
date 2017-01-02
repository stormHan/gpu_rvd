
#include <basic\process.h>
#include <basic\environment.h>
#include <basic\assert.h>
#include <basic\string.h>
#include <basic\stop_watch.h>


namespace {
	using namespace Gpu_Rvd;

	ThreadManager_var thread_manager_;
	int running_threads_invocations_ = 0;

	bool multithreading_initialized_ = false;
	bool multithreading_enabled_ = true;

	index_t max_threads_initialized_ = false;
	index_t max_threads_ = 0;

	bool fpe_initialized_ = false;
#ifdef GEO_DEBUG
	bool fpe_enabled_ = true;
#else
	bool fpe_enabled_ = false;
#endif

	bool cancel_initialized_ = false;
	bool cancel_enabled_ = false;

	double start_time_ = 0.0;

	/************************************************************************/
	/**
	* \brief Process Environment
	* \details This environment exposes and controls the configuration of the
	* Process module.
	*/
	class ProcessEnvironment : public Environment {
	protected:
		/**
		* \brief Gets a Process property
		* \details Retrieves the value of the property \p name and stores it
		* in \p value. The property must be a valid Process property (see
		* sys:xxx properties in Vorpaline's help).
		* \param[in] name name of the property
		* \param[out] value receives the value of the property
		* \retval true if the property is a valid Process property
		* \retval false otherwise
		* \see Environment::get_value()
		*/
		virtual bool get_local_value(
			const std::string& name, std::string& value
			) const {
			if (name == "sys:nb_cores") {
				value = String::to_string(Process::number_of_cores());
				return true;
			}
			if (name == "sys:multithread") {
				value = String::to_string(multithreading_enabled_);
				return true;
			}
			if (name == "sys:max_threads") {
				value = String::to_string(
					Process::maximum_concurrent_threads()
					);
				return true;
			}
			if (name == "sys:FPE") {
				value = String::to_string(fpe_enabled_);
				return true;
			}
			if (name == "sys:cancel") {
				value = String::to_string(cancel_enabled_);
				return true;
			}
			if (name == "sys:assert") {
				value = assert_mode() == ASSERT_THROW ? "throw" : "abort";
				return true;
			}
			return false;
		}

		/**
		* \brief Sets a Process property
		* \details Sets the property \p name with value \p value in the
		* Process. The property must be a valid Process property (see sys:xxx
		* properties in Vorpaline's help) and \p value must be a legal value
		* for the property.
		* \param[in] name name of the property
		* \param[in] value value of the property
		* \retval true if the property was sucessfully set
		* \retval false otherwise
		* \see Environment::set_value()
		*/
		virtual bool set_local_value(
			const std::string& name, const std::string& value
			) {
			if (name == "sys:multithread") {
				Process::enable_multithreading(String::to_bool(value));
				return true;
			}
			if (name == "sys:max_threads") {
				Process::set_max_threads(String::to_uint(value));
				return true;
			}
			if (name == "sys:FPE") {
				Process::enable_FPE(String::to_bool(value));
				return true;
			}
			if (name == "sys:cancel") {
				Process::enable_cancel(String::to_bool(value));
				return true;
			}
			if (name == "sys:assert") {
				if (value == "throw") {
					set_assert_mode(ASSERT_THROW);
					return true;
				}
				if (value == "abort") {
					set_assert_mode(ASSERT_ABORT);
					return true;
				}
				std::cerr << "Process " 
					<< "Invalid value for property sys:abort: "
					<< value
					<< std::endl;
				return false;
			}
			return false;
		}

		/** ProcessEnvironment destructor */
		virtual ~ProcessEnvironment() {
		}
	};
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

		if (Process::multithreading_enabled() && max_threads > 1) {
			run_concurrent_threads(threads, max_threads);
		}
		else {
			for (index_t i = 0; i < threads.size(); i++) {
				threads[i]->run();
			}
		}
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
	namespace Process {

		// OS dependent functions implemented in process_unix.cpp and
		// process_win.cpp

		bool os_init_threads();
		void os_brute_force_kill();
		bool os_enable_FPE(bool flag);
		bool os_enable_cancel(bool flag);
		void os_install_signal_handlers();
		index_t os_number_of_cores();
		size_t os_used_memory();
		size_t os_max_used_memory();
		std::string os_executable_filename();

		void initialize() {

			Environment* env = Environment::instance();
			env->add_environment(new ProcessEnvironment);

			if (!os_init_threads()) {
#ifdef GEO_OPENMP
				Logger::out("Process")
					<< "Using OpenMP threads"
					<< std::endl;
				set_thread_manager(new OMPThreadManager);
#else
				std::cout << "Process" 
					<< "Multithreading not supported, going monothread"
					<< std::endl;
				set_thread_manager(new MonoThreadingThreadManager);
#endif
			}

			os_install_signal_handlers();

			// Initialize Process default values

			enable_multithreading(multithreading_enabled_);
			set_max_threads(number_of_cores());
			enable_FPE(fpe_enabled_);
			enable_cancel(cancel_enabled_);

			start_time_ = SystemStopwatch::now();
		}

		void show_stats() {

			std::cout <<  "Process " << "Total elapsed time: "
				<< SystemStopwatch::now() - start_time_
				<< "s" << std::endl;

			const size_t K = size_t(1024);
			const size_t M = K*K;
			const size_t G = K*M;

			size_t max_mem = Process::max_used_memory();
			size_t r = max_mem;

			size_t mem_G = r / G;
			r = r % G;
			size_t mem_M = r / M;
			r = r % M;
			size_t mem_K = r / K;
			r = r % K;

			std::string s;
			if (mem_G != 0) {
				s += String::to_string(mem_G) + "G ";
			}
			if (mem_M != 0) {
				s += String::to_string(mem_M) + "M ";
			}
			if (mem_K != 0) {
				s += String::to_string(mem_K) + "K ";
			}
			if (r != 0) {
				s += String::to_string(r);
			}

			std::cout << ("Process") << "Maximum used memory: "
				<< max_mem << " (" << s << ")"
				<< std::endl;
		}

		void terminate() {
			thread_manager_.reset();
		}

		void brute_force_kill() {
			os_brute_force_kill();
		}

		index_t number_of_cores() {
			static index_t result = 0;
			if (result == 0) {
				result = os_number_of_cores();
			}
			return result;
		}

		size_t used_memory() {
			return os_used_memory();
		}

		size_t max_used_memory() {
			return os_max_used_memory();
		}

		std::string executable_filename() {
			return os_executable_filename();
		}

		void set_thread_manager(ThreadManager* thread_manager) {
			thread_manager_ = thread_manager;
		}

		void run_threads(ThreadGroup& threads) {
			running_threads_invocations_++;
			thread_manager_->run_threads(threads);
			running_threads_invocations_--;
		}

		void enter_critical_section() {
			thread_manager_->enter_critical_section();
		}

		void leave_critical_section() {
			thread_manager_->leave_critical_section();
		}

		bool is_running_threads() {
			return running_threads_invocations_ > 0;
		}

		bool multithreading_enabled() {
			return multithreading_enabled_;
		}

		void enable_multithreading(bool flag) {
			if (
				multithreading_initialized_ &&
				multithreading_enabled_ == flag
				) {
				return;
			}
			multithreading_initialized_ = true;
			multithreading_enabled_ = flag;
			if (multithreading_enabled_) {
				std::cout << ("Process")
					<< "Multithreading enabled" << std::endl
					<< "Available cores = " << number_of_cores()
					<< std::endl;
				// Logger::out("Process")
				//    << "Max. concurrent threads = "
				//    << maximum_concurrent_threads() << std::endl ;
				if (number_of_cores() == 1) {
					std::cerr << ("Process")
						<< "Processor is not a multicore"
						<< std::endl;
				}
				if (thread_manager_ == nil) {
					std::cerr << ("Process")
						<< "Missing multithreading manager"
						<< std::endl;
				}
			}
			else {
				std::cerr << ("Process")
					<< "Multithreading disabled" << std::endl;
			}
		}

		index_t max_threads() {
			return max_threads_initialized_
				? max_threads_
				: number_of_cores();
		}

		void set_max_threads(index_t num_threads) {
			if (
				max_threads_initialized_ &&
				max_threads_ == num_threads
				) {
				return;
			}
			max_threads_initialized_ = true;
			if (num_threads == 0) {
				num_threads = 1;
			}
			else if (num_threads > number_of_cores()) {
				std::cerr << ("Process")
					<< "Cannot allocate " << num_threads
					<< " for multithreading"
					<< std::endl;
				num_threads = number_of_cores();
			}
			max_threads_ = num_threads;
			std::cerr << ("Process")
				<< "Max used threads = " << max_threads_
				<< std::endl;
		}

		index_t maximum_concurrent_threads() {
			if (!multithreading_enabled_ || thread_manager_ == nil) {
				return 1;
			}
			return max_threads_;
			/*
			// commented out for now, since under Windows,
			// it seems that maximum_concurrent_threads() does not
			// report the number of hyperthreaded cores.
			return
			geo_min(
			thread_manager_->maximum_concurrent_threads(),
			max_threads_
			) ;
			*/
		}

		bool FPE_enabled() {
			return fpe_enabled_;
		}

		void enable_FPE(bool flag) {
			if (fpe_initialized_ && fpe_enabled_ == flag) {
				return;
			}
			fpe_initialized_ = true;
			fpe_enabled_ = flag;

			if (os_enable_FPE(flag)) {
				std::cerr << ("Process")
					<< (flag ? "FPE enabled" : "FPE disabled")
					<< std::endl;
			}
			else {
				std::cerr << ("Process")
					<< "FPE control not implemented" << std::endl;
			}
		}

		bool cancel_enabled() {
			return cancel_enabled_;
		}

		void enable_cancel(bool flag) {
			if (cancel_initialized_ && cancel_enabled_ == flag) {
				return;
			}
			cancel_initialized_ = true;
			cancel_enabled_ = flag;

			if (os_enable_cancel(flag)) {
				std::cerr << ("Process")
					<< (flag ? "Cancel mode enabled" : "Cancel mode disabled")
					<< std::endl;
			}
			else {
				std::cerr << ("Process")
					<< "Cancel mode not implemented" << std::endl;
			}
		}
	}
}