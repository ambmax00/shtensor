#include "ThreadBarrier.h"

namespace Shtensor
{

ThreadBarrier::ThreadBarrier(int _nb_threads) 
  : m_nb_threads(_nb_threads)
  , m_remaining(_nb_threads)
  , m_phase(0) 
{
}

void ThreadBarrier::wait() 
{
  std::unique_lock<std::mutex> lLock{m_mutex};
  auto gen = m_phase;
  if (!--m_remaining) 
  {
      m_phase++;
      m_remaining = m_nb_threads;
      m_barrier_condition.notify_all();
  } 
  else 
  {
      m_barrier_condition.wait(lLock, [this, gen] { return gen != m_phase; });
  }
}

} // end namespace