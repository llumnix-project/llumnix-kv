#pragma once


namespace blade_llm {

#ifdef NDEBUG
inline void fault_inject_throw() {}
inline void fault_inject_sleep(int) {}
#else
void fault_inject_throw();
void fault_inject_sleep(int dur_us);
#endif

}  // namespace blade_llm {