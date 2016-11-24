/*
	implemention of stopwath class
*/

#include <basic\stop_watch.h>

namespace Gpu_Rvd{

	StopWatch::StopWatch(std::string name){
		start_ = GetTickCount();
		name_ = name;
	}


	double StopWatch::elapsed_time() const{
		return double(GetTickCount() - start_) / 1000.0;
	}


	void StopWatch::print_elaspsed_time(std::ostream& os) const {
		os << "---- Times (seconds) ----\n"
			<< "---- Task :" << name_ << "----"
			<< "\n  Elapsed time: " << elapsed_time()
			<< std::endl;
	}

	StopWatch::~StopWatch(){

	}
}