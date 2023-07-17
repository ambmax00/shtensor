#ifndef SHTENSOR_THREADPOOL_H
#define SHTENSOR_THREADPOOL_H

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include "Logger.h"
#include "ThreadBarrier.h"

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

  void run(int64_t _start, int64_t _end, int64_t _step, LoopFunction&& _function);
 
 private:

  void thread_loop(int _id);

  int m_nb_threads;

  std::vector<std::thread> m_threads;

  std::mutex m_m2t_mutex;

  std::mutex m_t2m_mutex;

  std::condition_variable m_m2t_condition;

  std::condition_variable m_t2m_condition;

  int m_t2m_idx;

  std::vector<AtomicPtr<int64_t>> m_loop_p_index;

  std::vector<int64_t> m_loop_end;

  int64_t m_loop_step;

  LoopFunction m_loop_function;

  std::atomic<int> m_tasks_done_idx;

  bool m_stop;

  ThreadBarrier m_thread_barrier;

  Log::Logger m_logger;

};

} // end namespace Shtensor

#endif 