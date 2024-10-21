#include <stdexcept>
#include "naming.h"
#include "naming/shm_naming.h"
#include "logging.h"

namespace blade_llm {
static INamingService* NAMING = nullptr;

void connect_naming(const std::string &url) {
  auto pos = url.find(SCHEMA_DELIMITER);
  if (pos == std::string::npos) {
    throw std::runtime_error("unrecognized naming url;");
  }
  auto schema = url.substr(0, pos);
  auto content = url.substr(pos + 1, url.size());
  if (schema == SHARE_MEMORY_NAMING_SCHEMA) {
    NAMING = new ShmNaming();
  } else {
    // TODO : Parse naming schema and create naming client;
    LOG(ERROR) << "unknown naming schema: " << schema;
    throw std::runtime_error("unknown naming schema: " + schema);
  }
  NAMING->init(content);
}

INamingService* naming() {
  if (NAMING == nullptr) {
    throw std::runtime_error("naming service not connected;");
  }
  return NAMING;
}
}
