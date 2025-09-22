
#include "fault_inject.h"
#include "envcfg.h"
#include <random>
#include "thrid_party/logging.h"
#include <unistd.h>

namespace blade_llm {

#ifdef NDEBUG
void fault_inject_throw() {
}
#else
static constexpr uint64_t SCALE = 100000;
static thread_local std::mt19937_64 t_rng{(unsigned long) gettid()};
static thread_local std::uniform_int_distribution<uint64_t> t_random_dist{0, 100 * SCALE - 1};

void fault_inject_throw() {
  uint64_t rate = (env_debug_tx_failrate() - 1) * SCALE;
  uint64_t rnd = t_random_dist(t_rng);
  if (rnd >= rate) {
    return ;
  }
  throw std::runtime_error("fault inject");
  // this->rng_.seed(std::uint_fast64_t(s->dstworkerid_));
}
#endif

}  // namespace blade_llm {