#ifndef SHTENSOR_PROGRESS_THREAD
#define SHTENSOR_PROGRESS_THREAD

#include <chrono>
#include <mpi.h>
#include <thread>

#include "Logger.h"

namespace Shtensor 
{

class ProgressThread 
{
 public: 

  ProgressThread(MPI_Comm _comm)
    : m_comm(_comm)
    , m_rank(-1)
    , m_thread()
    , m_request(MPI_REQUEST_NULL)
    , m_terminate_flag(0)
    , m_wake_tag(666)
    , m_logger(Log::create("ProgressThread"))
  {
    MPI_Comm_rank(m_comm, &m_rank);
  }

  ProgressThread(const ProgressThread& _thread) = delete;

  ProgressThread(ProgressThread&& _thread) = default;

  ~ProgressThread()
  {
  }

  void start() 
  {
    MPI_Irecv(&m_terminate_flag, 1, MPI_INT, m_rank, m_wake_tag, m_comm, &m_request);
    m_thread = std::thread(&ProgressThread::progress_function, this);
  }

  void terminate()
  {
    MPI_Barrier(m_comm);

    int flag = 1;
    MPI_Send(&flag, 1, MPI_INT, m_rank, m_wake_tag, m_comm);
    m_thread.join();
  }

  void progress_function() 
  {
    Log::debug(m_logger, "Started progress thread");

    int flag = 0;
    int64_t rounds = 0;
    MPI_Status status; 

    while (1)
    {
      for (int i = 0; i < s_nb_tests; ++i)
      {
        MPI_Test(&m_request, &flag, &status);
      }

      if (flag)
      {
        break;
      }

      //std::this_thread::sleep_for(std::chrono::microseconds(10));
      std::this_thread::yield();

      ++rounds;
    }

    Log::debug(m_logger, "Exiting progress thread. Rounds: {}", rounds);
  }

 private:
  
  MPI_Comm m_comm;

  int m_rank;

  std::thread m_thread;

  MPI_Request m_request;

  int m_terminate_flag;

  const int m_wake_tag;

  Log::Logger m_logger;

  static inline constexpr int s_nb_tests = 1;

  static inline constexpr int s_sleep_ms = 0;

};

} // end namespace Shtensor

#endif