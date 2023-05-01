#ifndef SHTENSOR_LOGGER
#define SHTENSOR_LOGGER

#include <chrono>
#include "fmt/core.h"

namespace Shtensor::Log
{

#ifndef SHTENSOR_LOGLEVEL 
#define SHTENSOR_LOGLEVEL 1
#endif

constexpr static inline int g_loglevel = SHTENSOR_LOGLEVEL;

static inline std::FILE* g_stream = stdout;

class Logger
{
 public:

  Logger(const std::string& _name, std::FILE*& _p_stream)
    : m_p_stream(_p_stream)
    , m_name(_name)
  {
  }

  template <typename... Args>
  void print(const std::string& _msg, Args&&... _args)
  {
    fmt::print(m_p_stream, _msg, std::forward<Args>(_args)...);
  }

  const std::string& get_name()
  {
    return m_name;
  }

 private:

  std::FILE*& m_p_stream;

  std::string m_name;

};

static inline Logger g_logger = Logger("", g_stream);

static inline void set_global_stream(std::FILE* _pfile)
{
  g_stream = _pfile;
}

static inline Logger create(const std::string _name)
{
  Logger logger(_name, g_stream);
  return logger;
}

static inline std::string get_time()
{
  return "";
}

template <typename... Args>
static inline void info(Logger& _logger, const std::string _msg, Args&&... _args)
{
  if constexpr (g_loglevel > 1)
  {
   const std::string new_msg = fmt::format("[INFO] <{}> {}\n", _logger.get_name(), _msg);
  _logger.print(new_msg, std::forward<Args>(_args)...);
  }
}

template <typename... Args>
static inline void debug(Logger& _logger, const std::string _msg, Args&&... _args)
{
  if constexpr (g_loglevel > 2)
  {
   const std::string new_msg = fmt::format("[DEBUG] <{}> {}\n", _logger.get_name(), _msg);
  _logger.print(new_msg, std::forward<Args>(_args)...);
  }
}

template <typename... Args>
static inline void error(Logger& _logger, const std::string _msg, Args&&... _args)
{
  if constexpr (g_loglevel > 0)
  {
   const std::string new_msg = fmt::format("[ERROR] <{}> {}\n", _logger.get_name(), _msg);
  _logger.print(new_msg, std::forward<Args>(_args)...);
  }
}

} // end namespace Shtensor::Log

#endif // SHTENSOR_LOGGER