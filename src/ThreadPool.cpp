
#include "ThreadPool.h"
#include "Utils.h"

namespace Shtensor 
{

ThreadPool::ThreadPool(int _nb_threads)
  : m_nb_threads(_nb_threads)
  , m_threads(0)
  , m_run_condition()
  , m_loop_p_index()
  , m_loop_end()
  , m_loop_step()
  , m_thread_done_condition()
  , m_thread_done_idx(0)
  , m_loop_function()
  , m_stop(false)
  , m_logger(Log::create("ThreadPool"))
{
  if (m_nb_threads <= 0)
  {
    throw std::runtime_error("ThreadPool: Number of threads should be > 0");
  }

  std::generate_n(std::back_inserter(m_loop_p_index), m_nb_threads, 
                  []{ return std::make_unique<std::atomic<int64_t>>(); });

  m_loop_end.resize(m_nb_threads, 0);

  for (int i = 0; i < m_nb_threads; ++i)
  {
    m_threads.emplace_back(&ThreadPool::thread_loop, this, i);
  }
  
}

ThreadPool::~ThreadPool()
{
  {
    std::unique_lock<std::mutex> lock(m_run_mutex);
    m_stop = true;
  }
  m_run_condition.notify_all();

  for (auto& thread : m_threads)
  {
    thread.join();
  }
}

void ThreadPool::thread_loop(int _id)
{ 
  Log::debug(m_logger, "Hello from thread {}", _id);

  while (true)
  {
    {
      std::unique_lock<std::mutex> lock(m_run_mutex);
      m_run_condition.wait(lock, [this,_id]() { return m_stop || *m_loop_p_index[_id] < m_loop_end[_id]; });
    }

    // check for termination
    if (m_stop) break;

    // do work
    Log::debug(m_logger, "Thread {} starts working now", _id);
    while (*m_loop_p_index[_id] < m_loop_end[_id])
    {
      m_loop_function(_id);
      *m_loop_p_index[_id] += m_loop_step;
    }

    Log::debug(m_logger, "Thread {} is finished. Good work!", _id);
    // signal to main thread that we are done
    {
      std::unique_lock<std::mutex> lock(m_thread_done_mutex);
      m_thread_done_idx += 1;
      m_thread_done_condition.notify_one();
    }

  }

  Log::debug(m_logger, "Thread {} exiting", _id);
}

void ThreadPool::run(LoopFunction&& _function, int64_t _start, int64_t _end, int64_t _step)
{
  Log::debug(m_logger, "ThreadPool::run {}-->{}:{}", _start, _end, _step);

  int64_t nb_elements = _end - _start;

  // set up variables for running tasks
  {
    std::unique_lock<std::mutex> lock(m_run_mutex);

    m_loop_step = _step;

    m_loop_function = _function;

    for (int t = 0; t < m_nb_threads; ++t)
    {
      int64_t thread_start = 0;
      int64_t thread_end = 1;

      Utils::divide_equally(nb_elements, t, m_nb_threads, thread_start, thread_end);

      thread_start += _start;
      thread_end += _start;

      *m_loop_p_index[t] = thread_start;
      m_loop_end[t] = thread_end;
    }
  }

  m_run_condition.notify_all();
  
  // wait for threads to be done
  Log::debug(m_logger, "Main thread goes to sleep now. Goodnight.");
  {
    std::unique_lock<std::mutex> lock(m_thread_done_mutex);
    m_thread_done_condition.wait(lock, [this](){ return m_thread_done_idx == m_nb_threads; });
  }

  Log::debug(m_logger, "Good morning to main thread");
  m_thread_done_idx = 0;

}



} // end namespace Shtensor