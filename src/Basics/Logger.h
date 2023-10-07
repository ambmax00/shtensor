#ifndef SHTENSOR_LOGGER
#define SHTENSOR_LOGGER

#include <chrono>
#include "fmt/core.h"
#include "fmt/ranges.h"

namespace Shtensor::Log
{

#ifndef SHTENSOR_LOGLEVEL 
#define SHTENSOR_LOGLEVEL 1
#endif

constexpr static inline int g_loglevel = SHTENSOR_LOGLEVEL;

constexpr static inline int g_loglevel_critical = 1;
constexpr static inline int g_loglevel_error = 2;
constexpr static inline int g_loglevel_info = 3;
constexpr static inline int g_loglevel_debug = 4;

static inline std::FILE* g_stream = stdout;

class Logger
{
 public:

  Logger(const std::string& _name, std::FILE** _pp_stream)
    : m_pp_stream(_pp_stream)
    , m_name(_name)
  {
  }

  Logger(const Logger& _logger) = default;

  Logger(Logger&& _logger) = default;

  Logger& operator=(const Logger& _logger) = default; 

  Logger& operator=(Logger&& _logger) = default;

  template <typename... Args>
  void print(const std::string& _msg, Args&&... _args) const
  {
    fmt::print(*m_pp_stream, _msg, std::forward<Args>(_args)...);
  }

  const std::string& get_name() const
  {
    return m_name;
  }

 private:

  std::FILE** m_pp_stream;

  std::string m_name;

};

static inline Logger g_logger = Logger("", &g_stream);

static inline void set_global_stream(std::FILE* _pfile)
{
  g_stream = _pfile;
}

static inline Logger create(const std::string _name)
{
  Logger logger(_name, &g_stream);
  return logger;
}

static inline std::string get_time()
{
  return "";
}

template <typename... Args>
static inline void print(Logger& _logger, const std::string _msg, Args&&... _args)
{
  _logger.print(_msg, std::forward<Args>(_args)...);
}

template <typename... Args>
static inline void critical(Logger& _logger, const std::string _msg, Args&&... _args)
{
  if constexpr (g_loglevel >= g_loglevel_critical)
  {
   const std::string new_msg = fmt::format("[CRITICAL] <{}> {}\n", _logger.get_name(), _msg);
   _logger.print(new_msg, std::forward<Args>(_args)...);
  }
}

template <typename... Args>
static inline void info(Logger& _logger, const std::string _msg, Args&&... _args)
{
  if constexpr (g_loglevel >= g_loglevel_info)
  {
   const std::string new_msg = fmt::format("[INFO] <{}> {}\n", _logger.get_name(), _msg);
   _logger.print(new_msg, std::forward<Args>(_args)...);
  }
}

template <typename... Args>
static inline void debug(const Logger& _logger, const std::string _msg, Args&&... _args)
{
  if constexpr (g_loglevel >= g_loglevel_debug)
  {
    // Get the current time with milliseconds
    auto now = std::chrono::system_clock::now();
    
    // Extract hours, minutes, seconds, and milliseconds
    auto time_point = std::chrono::system_clock::to_time_t(now);
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000;

    std::tm time_info = *std::localtime(&time_point);
    
    int hours = time_info.tm_hour;
    int minutes = time_info.tm_min;
    int seconds = time_info.tm_sec;
    int ms = milliseconds.count();
    
    // Format the time to include milliseconds using fmt
    std::string formatted_time = fmt::format("{:02d}:{:02d}:{:02d}:{:03d}", 
                                             hours, minutes, seconds, ms);
    
   const std::string new_msg = fmt::format("[DEBUG] {} <{}> {}\n", formatted_time, 
    _logger.get_name(), _msg);

   _logger.print(new_msg, std::forward<Args>(_args)...);
  }
}

template <typename... Args>
static inline void error(Logger& _logger, const std::string _msg, Args&&... _args)
{
  if constexpr (g_loglevel >= g_loglevel_error)
  {
   const std::string new_msg = fmt::format("[ERROR] <{}> {}\n", _logger.get_name(), _msg);
   _logger.print(new_msg, std::forward<Args>(_args)...);
  }
}

#define DEBUG_VAR(logger, var) \
  debug(logger, "{}: {}", #var, var);

} // end namespace Shtensor::Log

#endif // SHTENSOR_LOGGER