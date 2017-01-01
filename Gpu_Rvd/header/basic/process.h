/**
 * \file header/basic/process.h
 * \basic Function and classes for process manipulation
 */

#include <basic\common.h>
#include <basic\counted.h>
#include <basic\smart_pointer.h>

namespace Gpu_Rvd{

#define GEO_THREAD_LOCAL __declspec(thread)

	/**
	 * \brief Platform-independent base class for running threads
	 * \details
	 * A Thread object manages one thread of control within the program.
	 * Thread%s begin executing with run(). Operational threads can be created
	 * by creating a derived class and reimplement function run().
	 *
	 * Thread%s are reference-counted objects. Their allocation and
	 * destruction can be automatically managed with Thread_var.
	 */
	class Thread : public Counted{
	public:
		/**
		* \brief Thread constructor.
		*/
		Thread() : id_(0) {
		}

		/**
		* \brief Starts the thread execution.
		*/
		virtual void run() = 0;

		/**
		* \brief Gets the identifier of this thread.
		* \return the identifier of the thread, i.e.
		*  an unsigned integer in the range [0, N-1]
		*  where N denotes the number of currently
		*  running threads.
		*/
		index_t id() const {
			return id_;
		}

		/**
		* \brief Gets the current thread.
		* \return A pointer to the instance of the
		*  currently running thread.
		*/
		static Thread* current();

	protected:
		/** Thread destructor */
		virtual ~Thread();


	private:
		/**
		* \brief Sets the indentifier of this thread.
		* \details This function is meant to be called
		*  by the thread manager for each created thread.
		* \param[in] id_in the identifier of this thread.
		*/
		void set_id(index_t id_in) {
			id_ = id_in;
		}

		/**
		* \brief Specifies the current instance, used by current().
		* \details Stores the specified thread in the thread-local-storage
		*   static variable so that current() can retreive it.
		*   Should be called by the ThreadManager right before launching
		*   the threads.
		* \param[in] thread a pointer to the thread currently executed
		*/
		static void set_current(Thread* thread);

		index_t id_;

		// ThreadManager needs to access set_current() and 
		// set_id().
		friend class ThreadManager;
	};

	/* Smart pointer that contains a Thread objecet */
	typedef SmartPointer<Thread> Thread_var;

	/**
	* \brief Collection of Thread%s
	* \details ThreadGroup is a std::vector of Thread_var it provides the
	* same operations for adding, removing or accessing thread elements.
	* ThreadGroup takes ownership of Thread elements when they are added to
	* the group, so there's is no need to delete Threads when the group is
	* deleted.
	*/
	typedef std::vector<Thread_var> ThreadGroup;

	/**
	* \brief Typed collection of Thread%s.
	* \details
	* TypedThreadGroup is a ThreadGroup that provides a typed accessor with
	* operator[]().
	* \tparam THREAD the type of Thread%s in the collection
	*/
	template <class THREAD>
	class TypedThreadGroup : public ThreadGroup {
	public:
		/**
		* \brief Creates an empty group of Thread%s
		* \details Thread elements can be added with the std::vector
		* operation push_back()
		*/
		TypedThreadGroup() {
		}

		/**
		* \brief Gets a thread element by index
		* \param[in] i index of the element
		* \return a pointer to the \p THREAD at position \p i in the
		* thread group
		*/
		THREAD* operator[] (index_t i) {
			geo_debug_assert(i < size());
			Thread* result = ThreadGroup::operator[] (i);
			return static_cast<THREAD*>(result);
		}
	};

	/**
	* \brief Platform-independent base class for running concurrent threads.
	* \details
	* The ThreadManager manager provides a platform-independent abstract
	* interface for running concurrent Threads and managing critical
	* sections.
	*
	* The ThreadManager is derived in multiple platform-specific or
	* technology-specific implementations.
	*
	* Platform-specific implementations:
	* - POSIX Thread manager (Unix)
	* - Windows Threads manager (Windows)
	* - Windows ThreadPool manager (Windows)
	*
	* Technology-specific implementations:
	* - OpenMP-based manager
	*
	* Which ThreadManager to use is determined at runtime by
	* Process::initialize() according to the current platform or the current
	* available technology.
	*
	* \note For internal use only.
	* \see Process::set_thread_manager()
	*/
	class ThreadManager : public Counted{
	public:
		/**
		* \brief Runs a group of Thread%s.
		* \details
		* This start the execution of the threads
		* contained in vector \p threads.
		*
		* If the threads cannot be executed in a concurrent environment
		* (multi-threading is disabled or the number of maximum threads is 1),
		* then the threads are executed sequentially. Otherwise the function
		* run_concurrent_threads() is called to execute the threads
		* concurrently. The execution terminates when the last thread
		* terminates.
		*
		* \param[in] threads the vector of threads to be executed.
		* \see maximum_concurrent_threads()
		* \see run_concurrent_threads()
		* \see Process::max_threads()
		*/
		virtual void run_threads(ThreadGroup& threads);

		/**
		* \brief Gets the maximum number of possible concurrent threads
		* \return The maximum number of possible concurrent threads allowed
		* by this manager. It depends on the physical number of cores
		* (including hyper-threading or not) and the technology implemented
		* by this manager.
		* \see Process::number_of_cores()
		*/
		virtual index_t maximum_concurrent_threads() = 0;

		/**
		* \brief Enters a critical section
		* \details
		* One thread at a time can enter the critical section, all the other
		* threads that call this function are blocked until the blocking
		* thread leaves the critical section.
		* \see leave_critical_section()
		*/
		virtual void enter_critical_section() = 0;

		/**
		* \brief Leaves a critical section
		* \details When a blocking thread leaves a critical section, this
		* makes the critical section available for a waiting thread.
		* \see enter_critical_section()
		*/
		virtual void leave_critical_section() = 0;

	protected:
		/**
		* \brief Runs a group of Thread%s concurrently.
		* \details This start the concurrent execution of the threads
		* contained in vector \p threads, using the given number of threads
		* \p max_threads. The execution terminates when the last thread
		* terminates.
		* \param[in] threads the vector of threads to be executed.
		* \param[in] max_threads maximum number of threads allowed for this
		* execution. It is always greater than one
		*/
		virtual void run_concurrent_threads(
			ThreadGroup& threads, index_t max_threads
			) = 0;
		/**
		* \brief Sets the id of a thread.
		* \details This function is called right before starting
		*  the threads. Each thread will have an id in [0, N-1]
		*  where N denotes the number of running threads.
		* \param[in] thread the thread
		* \param[in] id the id
		*/

		static void set_thread_id(Thread* thread, index_t id) {
			thread->set_id(id);
		}

		/**
		* \brief Specifies the current instance, used by current().
		* \details Stores the specified thread in the thread-local-storage
		*   static variable so that current() can retreive it.
		*   Should be called by the ThreadManager right before launching
		*   the threads.
		* \param[in] thread a pointer to the thread currently executed
		*/
		static void set_current_thread(Thread* thread) {
			Thread::set_current(thread);
		}

		/** ThreadManager destructor */
		virtual ~ThreadManager();
	};
	/** Smart pointer that contains a ThreadManager object */
	typedef SmartPointer<ThreadManager> ThreadManager_var;

	/**
	* \brief Single thread ThreadManager
	* \details MonoThreadingThreadManager implements a ThreadManager for
	* single thread environments.
	*/

	class MonoThreadingThreadManager : public ThreadManager{
	public:
		/**
		* \copydoc ThreadManager::maximum_concurrent_threads()
		* \note This implementation always returns 1.
		*/
		virtual index_t maximum_concurrent_threads(){
			return index_t(1);
		}

		/**
		* \copydoc ThreadManager::enter_critical_section()
		* \note This implementation does actually nothing
		*/
		virtual void enter_critical_section();

		/**
		* \copydoc ThreadManager::leave_critical_section()
		* \note This implementation does actually nothing
		*/
		virtual void leave_critical_section();

	protected:
		/** MonoThreadingThreadManager destructor */
		virtual ~MonoThreadingThreadManager();

		/**
		* \copydoc ThreadManager::run_concurrent_threads()
		* \note This implementation always executes threads sequentially.
		*/
		virtual void run_concurrent_threads(
			ThreadGroup& threads, index_t max_threads
			);
	};
}
