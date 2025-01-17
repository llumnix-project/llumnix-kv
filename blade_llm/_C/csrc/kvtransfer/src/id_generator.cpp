#include "utils/id_generator.h"
#include <atomic>

namespace blade_llm {
uint64_t new_id() noexcept {
  // reserved: [0, 20181218)
  static std::atomic<uint64_t> last_id_ = 20181218;
  auto v = last_id_.fetch_add(1, std::memory_order_relaxed);
  return v;
}

}  // namespace blade_llm {