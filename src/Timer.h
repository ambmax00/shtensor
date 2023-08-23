#ifndef SHTENSOR_TIMER_H
#define SHTENSOR_TIMER_H

#include <chrono>

namespace Shtensor 
{

class Timer
{
 public:

  Timer();

  double elapsed();

 private:

  std::chrono::time_point<std::chrono::steady_clock> m_start_time;
  
};

}

#endif // SHTENSOR_TIMER_H