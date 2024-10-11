#include <stdexcept>
#include "naming.h"

namespace blade_llm {

static INamingService* NAMING = nullptr;

void connect_naming(const std::string &url) {
  auto pos = url.find(SCHEMA_DELIMITER);
  if (pos == std::string::npos) {
    throw std::runtime_error("unrecognized naming url;");
  }
  auto schema = url.substr(0, pos);
  auto content = url.substr(pos + 1, url.size());
  // TODO : Parse naming schema and create naming client;
  throw std::runtime_error("unknown naming schema: " + schema);
}

INamingService* naming() {
  if (NAMING == nullptr) {
    throw std::runtime_error("naming service not connected;");
  }
  return NAMING;
}
}
