#ifndef KVTRANSFER_INCLUDE_UTILS_HTTP_CLIENT_H_
#define KVTRANSFER_INCLUDE_UTILS_HTTP_CLIENT_H_
#include <string>
#include <mutex>
#include <optional>

namespace blade_llm {

class IHttpClient {
 public:
  virtual void post(const std::string& url, const std::string& data) = 0;
  virtual std::optional<std::string> get(const std::string& url) = 0;
  virtual ~IHttpClient() = default;
};

class CurlHttpClient: public IHttpClient {
 public:
  CurlHttpClient();
  void post(const std::string& url, const std::string& data) override;
  std::optional<std::string> get(const std::string& url) override;
};
}
#endif //KVTRANSFER_INCLUDE_UTILS_HTTP_CLIENT_H_
