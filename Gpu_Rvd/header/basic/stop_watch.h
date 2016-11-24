/*
StopWatch:
to record the tatol time in Cpu + Gpu
*/

#ifndef BASIC_STOPWATCH_H
#define BASIC_STOPWATCH_H

#include <basic\common.h>

#include <Windows.h>

namespace Gpu_Rvd{

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