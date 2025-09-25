#pragma once

#include <stdlib.h>
#include <memory>

namespace blade_llm {

class GDRMHandle;
class MmapHandle;

class GdrMemDesc {
  void* const cpu_ptr_ = nullptr;
  std::unique_ptr<GDRMHandle> handle_;
  std::unique_ptr<MmapHandle> mmhandle_;
public:
  GdrMemDesc(void* cpu_ptr, std::unique_ptr<GDRMHandle> handle,
             std::unique_ptr<MmapHandle> mmhandle) noexcept;

  ~GdrMemDesc();

  void* cpu_ptr() const noexcept {
    return this->cpu_ptr_;
  }

private:
  GdrMemDesc(const GdrMemDesc&) = delete;
  GdrMemDesc(GdrMemDesc&&) = delete;
};

// RETUREN.cpu_ptr <-> gpu_ptr
// null means bad desc
std::unique_ptr<GdrMemDesc> gdrcpy_mmap(void* gpu_ptr, size_t size) noexcept;

}  // namespace blade_llm {
