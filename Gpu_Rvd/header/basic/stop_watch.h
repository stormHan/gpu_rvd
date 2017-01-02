/*
StopWatch:
to record the tatol time in Cpu + Gpu
*/

#ifndef BASIC_STOPWATCH_H
#define BASIC_STOPWATCH_H

#include <basic\common.h>

#include <Windows.h>

namespace Gpu_Rvd{

	/**
	* \brief Measures the time taken by an algorithm.
	* \details
	* SystemStopwatch provides functions to get or print the time
	* elapsed since its construction. The times computed by
	* SystemStopwatch are expressed as system ticks, which is a system
	* dependant unit. SystemStopwatch prints three different times:
	*
	* - real time: the really elapsed time (depends on the load of the
	*   machine, i.e. on the others programs that are executed at the
	*   same time).
	* - system time: the time spent in system calls.
	* - user time: the time really spent in the process.
	*
	* Example:
	* \code
	* {
	*     SystemStopwatch clock ;
	*     do_something() ;
	*     clock.print_elapsed_time(std::cout) ;
	* }
	* \endcode
	*
	*/

	class  SystemStopwatch {
	public:
		/**
		* \brief SystemStopwatch constructor
		* \details It remembers the current time as the reference time
		* for functions elapsed_user_time() and print_elapsed_time().
		*/
		SystemStopwatch();

		/**
		* \brief Prints elapsed time to a stream
		* \details Prints real, user and system times since the
		* construction of this SystemStopWatch (in seconds).
		*/
		void print_elapsed_time(std::ostream& os) const;

		/**
		* \brief Get the user elapsed time
		* \details Returns the user time elapsed since the SystemStopWatch
		* construction (in seconds)
		*/
		double elapsed_user_time() const;

		/**
		* \details Gets the current time (in seconds).
		*/
		static double now();

	private:
#ifdef GEO_OS_WINDOWS
		long start_;
#else
		tms start_;
		clock_t start_user_;
#endif
	};

	class StopWatch{
	public:

		/*
			constructor function of StopWatch
		*/
		StopWatch(std::string name);

		/*
			get the current time;
		*/
		double now() const { return (double)(GetTickCount() / 1000.0); }

		/*
			get current elaspsed time.
		*/
		double elapsed_time() const;

		/*
			print elapsed time to a stream
		*/
		void print_elaspsed_time(std::ostream& os) const;

		/*
			destructor of stopwatch.
			print the time.
		*/
		~StopWatch();

	private:
		long start_;
		std::string name_;
	};
}

#endif /* BASIC_STOPWATCH_H */