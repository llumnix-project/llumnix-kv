
#ifndef KVTRANSFER_INCLUDE_ERROR_H_
#define KVTRANSFER_INCLUDE_ERROR_H_
#pragma once
#include <string>
#include <optional>

namespace blade_llm {
enum ErrorCode {
  SUCCESS,
  // connection related
  INVALID_TARGET,
  TARGET_NOT_FOUND,
  TARGET_DISCONNECTED,
  // request related
  REQUEST_NOT_FOUND,
  INVALID_REQUEST_PARAM,
  UNEXPECTED_REQ_RECV,
  //
  INVALID_OPERATION,
  UNKNOWN_FAILURE,
};

struct Error {
  ErrorCode code;
  std::string msg;

  Error(ErrorCode code, const std::string &msg) : code(code), msg(msg) {}
  Error(Error &&other) noexcept: code(other.code), msg(std::move(other.msg)) {}
  std::string to_string() const noexcept {
    return std::string("ErrorCode: ") + std::to_string(code) + ", msg: " + msg;
  }
};

template<typename T>
class Result {
 public:
  Result(Error &&err) : ok_(std::nullopt), error_(std::move(err)) {}
  Result(T &&ok) : ok_(std::move(ok)), error_(Error(SUCCESS, "")) {}

  static Result<T> error(ErrorCode code, const std::string &msg) {
    return Result<T>(Error(code, msg));
  }

  bool is_ok() const { return ok_.has_value(); }
  bool is_err() const { return !ok_.has_value(); }
  T &ok() { return ok_.value(); }
  Error &err() { return error_; }
 private:
  std::optional<T> ok_;
  Error error_;
};
}
#endif //KVTRANSFER_INCLUDE_ERROR_H_
