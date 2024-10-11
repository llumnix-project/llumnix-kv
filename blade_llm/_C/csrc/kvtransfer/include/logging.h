#ifndef KVTRANSFER_INCLUDE_LOGGING_H_
#define KVTRANSFER_INCLUDE_LOGGING_H_

// copy from: https://github.com/alibaba/BladeDISC/blob/main/tao_compiler/mlir/ral/ral_logging.h
#include <atomic>
#include <cassert>
#include <limits>
#include <memory>
#include <sstream>

namespace blade_llm {
namespace ral {

const int INFO = 0;            // base_logging::INFO;
const int WARNING = 1;         // base_logging::WARNING;
const int ERROR = 2;           // base_logging::ERROR;
const int FATAL = 3;           // base_logging::FATAL;
const int NUM_SEVERITIES = 4;  // base_logging::NUM_SEVERITIES;

namespace internal {

#define PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char *fname, int line, int severity);
  ~LogMessage() override;

  // Change the location of the log message.
  LogMessage &AtLocation(const char *fname, int line);

  // Returns the minimum log level for VLOG statements.
  // E.g., if MinVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int MinVLogLevel();

  // Returns whether VLOG level lvl is activated for the file fname.
  //
  // E.g. if the environment variable TF_CPP_VMODULE contains foo=3 and fname is
  // foo.cc and lvl is <= 3, this will return true. It will also return true if
  // the level is lower or equal to TF_CPP_MIN_VLOG_LEVEL (default zero).
  //
  // It is expected that the result of this query will be cached in the VLOG-ing
  // call site to avoid repeated lookups. This routine performs a hash-map
  // access against the VLOG-ing specification provided by the env var.
  static bool VmoduleActivated(const char *fname, int level);

 protected:
  void GenerateLogMessage();

 private:
  const char *fname_;
  int line_;
  int severity_;
};

// Uses the lower operator & precedence to voidify a LogMessage reference, so
// that the ternary VLOG() implementation is balanced, type wise.
struct Voidifier {
  // clang-format off
  template<typename T>
  void operator&(const T &) const {}
  // clang-format on
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char *file, int line);
  ~LogMessageFatal() override;
};

// LogMessageNull supports the DVLOG macro by simply dropping any log messages.
class LogMessageNull : public std::basic_ostringstream<char> {
 public:
  LogMessageNull() {}
  ~LogMessageNull() override {}
};

}  // namespace internal

#define _LOG_INFO \
  ::blade_llm::ral::internal::LogMessage(__FILE__, __LINE__, ::blade_llm::ral::INFO)
#define _LOG_WARNING \
  ::blade_llm::ral::internal::LogMessage(__FILE__, __LINE__, ::blade_llm::ral::WARNING)
#define _LOG_ERROR \
  ::blade_llm::ral::internal::LogMessage(__FILE__, __LINE__, ::blade_llm::ral::ERROR)
#define _LOG_FATAL ::blade_llm::ral::internal::LogMessageFatal(__FILE__, __LINE__)

#define LOG(severity) _LOG_##severity

#define VLOG_IS_ON(lvl)                                               \
  (([](int level, const char* fname) {                                    \
    static const bool vmodule_activated =                                 \
        ::blade_llm::ral::internal::LogMessage::VmoduleActivated(fname, level); \
    return vmodule_activated;                                             \
  })(lvl, __FILE__))

#define VLOG(level)                                        \
  PREDICT_TRUE(!VLOG_IS_ON(level))                     \
  ? (void)0                                                    \
  : ::blade_llm::ral::internal::Voidifier() &                        \
          ::blade_llm::ral::internal::LogMessage(__FILE__, __LINE__, \
                                           ::blade_llm::ral::INFO)

#define CHECK(expr) \
  while (!(expr)) LOG(FATAL) << "CHECK failed: "

}  // namespace ral
}  // namespace blade_llm

#endif //KVTRANSFER_INCLUDE_LOGGING_H_
