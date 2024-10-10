#ifndef KVTRANSFER_INCLUDE_UTILS_ITERATOR_H_
#define KVTRANSFER_INCLUDE_UTILS_ITERATOR_H_

#pragma once
#include <optional>
#include <functional>
#include <memory>

namespace blade_llm {

template<typename T>
class IIterator {
 public:
  virtual std::optional<T> next() = 0;
  virtual ~IIterator() = default;
};

template<typename T>
class Source : public IIterator<T *> {
 public:

  static std::unique_ptr<IIterator<T *>> from(std::vector<T> &src) {
    return std::make_unique<Source<T>>(src.data(), src.size());
  };

  static std::unique_ptr<IIterator<T *>> from(T *src, size_t nums) {
    return std::make_unique<Source<T>>(src, nums);
  };

  Source(T *src, size_t num_elements) :
      src_(src),
      num_elements_(num_elements) {};

  std::optional<T *> next() override {
    if (cursor < num_elements_) {
      return &src_[cursor++];
    } else {
      return std::nullopt;
    }
  }
 private:
  T *src_;
  size_t num_elements_;
  size_t cursor{0};
};

template<typename T>
class CopySource : public IIterator<T> {
 public:
  static std::unique_ptr<IIterator<T>> from(const std::vector<T> &src) {
    return std::make_unique<CopySource<T>>(src.data(), src.size());
  };

  CopySource(const T *src, size_t num_elements) :
      src_(src),
      num_elements_(num_elements) {};

  std::optional<T> next() override {
    if (cursor < num_elements_) {
      return src_[cursor++];
    } else {
      return std::nullopt;
    }
  }
 private:
  const T *src_;
  size_t num_elements_;
  size_t cursor{0};
};

template<typename T>
class Filter : public IIterator<T> {
 public:
  static std::unique_ptr<IIterator<T>> from(std::unique_ptr<IIterator<T>> &src,
                                            std::function<bool(const T &)> filter) {
    return std::make_unique<Filter<T>>(src, filter);
  };

  Filter(std::unique_ptr<IIterator<T>> &src, std::function<bool(const T &)> filter) :
      src_(std::move(src)),
      filter_(filter) {};

  std::optional<T> next() override {
    for (;;) {
      auto src_next = src_->next();
      if (src_next.has_value()) {
        if (filter_(src_next.value())) {
          return src_next;
        }
      } else {
        return std::nullopt;
      }
    }
  }

 private:
  std::unique_ptr<IIterator<T>> src_;
  std::function<bool(const T &)> filter_;
};

template<typename T, typename R>
class Map : public IIterator<R> {
 public:
  Map(std::unique_ptr<IIterator<T>> &src, std::function<R(const T &)> map) :
      src_(std::move(src)),
      map_(map) {};

  std::optional<R> next() override {
    auto src_next = src_->next();
    if (src_next.has_value()) {
      return map_(src_next.value());
    } else {
      return std::nullopt;
    }
  }

 private:
  std::unique_ptr<IIterator<T>> src_;
  std::function<R(const T &)> map_;
};

template<typename T>
class Iterator : public IIterator<T> {
 public:

  static Iterator<T> copy_from(const std::vector<T> &src) {
    return Iterator<T>(std::make_unique<CopySource<T>>(src.data(), src.size()));
  }

  explicit Iterator(std::unique_ptr<IIterator<T>> &&iter) : iter_(std::move(iter)) {};

  Iterator(const Iterator &other) = delete;
  Iterator &operator=(const Iterator &other) = delete;

  Iterator(Iterator &&other) noexcept: iter_(std::move(other.iter_)) {};
  Iterator &operator=(Iterator &&other) noexcept {
    if (this != &other) {
      iter_ = std::move(other.iter_);
    }
    return *this;
  }

  bool has_next() {
    if (next_value_.has_value()) {
      return true;
    }
    if (iter_ == nullptr) {
      return false;
    }
    next_value_ = iter_->next();
    return next_value_.has_value();
  }

  std::optional<T> next() override {
    if (next_value_.has_value()) {
      auto r = next_value_;
      next_value_.reset();
      return r;
    }
    if (iter_ == nullptr) {
      return std::nullopt;
    }
    return iter_->next();
  }

  Iterator<T> filter(std::function<bool(const T &)> f) {
    return Iterator<T>(std::make_unique<Filter<T>>(iter_, f));
  }

  template<class R>
  std::unique_ptr<IIterator<R>> map(std::function<R(const T &)> f) {
    return std::make_unique<Map<T, R>>(iter_, f);
  }

 private:
  std::optional<T> next_value_{std::nullopt};
  std::unique_ptr<IIterator<T>> iter_;
};
} // namespace blade_llm
#endif //KVTRANSFER_INCLUDE_UTILS_ITERATOR_H_
