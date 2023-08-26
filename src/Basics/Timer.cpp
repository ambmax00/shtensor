#include "Timer.h"

namespace Shtensor
{

Timer::Timer() : m_start_time(std::chrono::steady_clock::now())
{ 
}

double Timer::elapsed()
{
  const auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - m_start_time;
  return elapsed_seconds.count();
}

}