
#ifndef KVTRANSFER_INCLUDE_ERROR_H_
#define KVTRANSFER_INCLUDE_ERROR_H_
#pragma once
#include <string>
#include <optional>
#include <stdexcept>

namespace blade_llm {

enum class ErrorKind: unsigned int {
  OTHER_ERROR = 0,
  // connection related
  INVALID_TARGET = 1,
  TARGET_NOT_FOUND = 2,
  TARGET_CONNOT_CONNECT = 3,
  TARGET_DISCONNECTED = 4,
  // request related
  REQUEST_NOT_FOUND = 5,
  INVALID_REQUEST_PARAM = 6,
  UNEXPECTED_REQ_RECV = 7,
  //
  INVALID_OPERATION = 8,
};

static const char *ERROR_NAMES[] = {"Other error",
                                    "Invalid target",
                                    "Target not found",
                                    "Target cannot connect",
                                    "Target disconnected",
                                    "Request not found",
                                    "Invalid request",
                                    "Unexpected request",
                                    "Invalid operation"};

class KVTransferException : public std::exception {
 public:
  KVTransferException(ErrorKind kind, const std::string &msg) :
      msg_(ERROR_NAMES[static_cast<unsigned int>(kind)]) {
    if (!msg.empty()) {
      msg_ += ": " + msg;
    }
  }
  const char *what() const noexcept override {
    return msg_.c_str();
  }
 private:
  std::string msg_;
};

}
#endif //KVTRANSFER_INCLUDE_ERROR_H_
