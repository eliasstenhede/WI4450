
#include "timer.hpp"

#include <iomanip>

// for timing routine
#include <omp.h>

#include <iostream>
#include <iomanip>

// static members of a class must be defined
// somewhere in an object file, otherwise you
// will get linker errors (undefined reference)
std::map<std::string, int> Timer::counts_;
std::map<std::string, double> Timer::flops_;
std::map<std::string, double> Timer::bytes_;
std::map<std::string, double> Timer::times_;

  Timer::Timer(std::string label, double flops, double bytes)
  : label_(label)
  {
    t_start_ = omp_get_wtime();
    flops_[label_] = flops;
    bytes_[label_] = bytes;
  }


  Timer::~Timer()
  {
    double t_end = omp_get_wtime();
    times_[label_] += t_end - t_start_;
    counts_[label_]++;
  }

void Timer::summarize(std::ostream& os)
{
  os << "==================== TIMER SUMMARY ==============================================================" << std::endl;
  os << "         label         calls    total time     mean time     intensity       Gflop/s      Gbyte/s" << std::endl;
  os << "-------------------------------------------------------------------------------------------------" << std::endl;
  for (auto [label, time]: times_)
  {
    int count = counts_[label];
    float gflops = flops_[label];
    float gbytes = bytes_[label];
    std::cout << std::setw(14) << label;
		std::cout << std::setw(14) << count;
		std::cout << std::setw(14) << time;
		std::cout << std::setw(14) << time/double(count);
		std::cout << std::setw(14) << gflops/gbytes;
		std::cout << std::setw(14) << gflops/time*double(count);
		std::cout << std::setw(13) << gbytes/time*double(count);
		std::cout << std::endl;
  }
  os << "=================================================================================================" << std::endl;
}
