#ifndef SHTENSOR_THREADPOOL_H
#define SHTENSOR_THREADPOOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "Logger.h"

namespace Shtensor
{

class ThreadPool
{
 public:

  using LoopFunction = std::function<void(int64_t)>;

  template <typename T>
  using AtomicPtr = std::unique_ptr<std::atomic<T>>;

  ThreadPool(int _nb_threads = static_cast<int>(std::thread::hardware_concurrency()));

  ~ThreadPool();

  void run(LoopFunction&& _function, int64_t _start, int64_t _end, int64_t _step = 1);
 
 private:

  void thread_loop(int _id);

  int m_nb_threads;

  std::vector<std::thread> m_threads;

  std::mutex m_run_mutex;

  std::condition_variable m_run_condition;

  LoopFunction m_loopFunction;

  std::vector<AtomicPtr<int64_t>> m_loop_p_index;

  std::vector<int64_t> m_loop_end;

  int64_t m_loop_step;

  std::mutex m_thread_done_mutex;

  std::condition_variable m_thread_done_condition;

  int m_thread_done_idx;

  LoopFunction m_loop_function;

  volatile bool m_stop;

  Log::Logger m_logger;

};

} // end namespace Shtensor

#endif 