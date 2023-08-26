#ifndef SHTENSOR_THREADBARRIER_H
#define SHTENSOR_THREADBARRIER_H

#include <condition_variable>
#include <mutex>

namespace Shtensor 
{

class ThreadBarrier
{
 public:

  explicit ThreadBarrier(int _nb_threads);
 
  void wait();
 
 private:

  std::mutex m_mutex;
  std::condition_variable m_barrier_condition;
  int m_nb_threads;
  int m_remaining;
  int m_phase;

};

} // end namespace Shtensor

#endif 