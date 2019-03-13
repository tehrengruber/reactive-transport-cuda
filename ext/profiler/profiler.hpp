#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <stdint.h>
#include <algorithm>  // min, max
#include <cassert>
#include <cstddef>  // size_t
#include <cstring>
#include <iostream>
#include <limits>

#ifdef PROFILER_USE_PCM
#include "pcm/cpucounters.h"
#endif

#ifndef NDEBUG
#include <stack>
#endif

// configuration
#define PROFILER_MAX_DEPTH 10
#define PROFILER_MAX_SLOTS 100
#define PROFILER_TIMING_BIN_SIZE 10
#define PROFILER_MAX_KEY_LENGTH 100

// profiling macros
#ifdef SKIP_PROFILING
#define PROFILER_START(key)
#define PROFILER_STOP(key)
#else
#ifdef PROFILER_USE_PCM
#define PROFILER_START(key)                                     \
  {                                                             \
    static size_t key_hash = Profiler::hash(key);               \
    Profiler::preprocess(key_hash);                             \
  }                                                             \
  asm volatile("" : : : "memory");                              \
  Profiler::msr_handle->read(CPU_CLK_UNHALTED_THREAD_ADDR,      \
                              &Profiler::start_time);           \
  asm volatile("" : : : "memory");

#define PROFILER_STOP(key)                                      \
  asm volatile("" : : : "memory");                              \
  Profiler::msr_handle->read(CPU_CLK_UNHALTED_THREAD_ADDR,      \
                              &Profiler::stop_time);            \
  asm volatile("" : : : "memory");                              \
  {                                                             \
    static Profiler::Timing* slot =                             \
        Profiler::initialize_timing(key, Profiler::chash(key)); \
    Profiler::postprocess(Profiler::chash(key), slot);          \
  }
#else
#define PROFILER_START(key)                                    \
  {                                                            \
    static size_t key_hash = Profiler::hash(key);              \
    Profiler::preprocess(key_hash);                            \
  }                                                            \
  asm volatile("" : : : "memory");                             \
  asm volatile("cpuid"                                         \
               :                                               \
               : "a"(0)                                        \
               : "bx", "cx", "dx"); /* CPUID exec-barrier */   \
  asm volatile("rdtsc"                                         \
               : "=a"((Profiler::start_time).lo),              \
                 "=d"((Profiler::start_time).hi)); /* RDTSC */ \
  asm volatile("" : : : "memory");

#define PROFILER_STOP(key)                                      \
  asm volatile("" : : : "memory");                              \
  asm volatile("rdtscp"                                         \
               : "=a"((Profiler::stop_time).lo),                \
                 "=d"((Profiler::stop_time).hi)); /* RDTSC */   \
  asm volatile("cpuid"                                          \
               :                                                \
               : "a"(0)                                         \
               : "bx", "cx", "dx"); /* CPUID exec-barrier */    \
  asm volatile("" : : : "memory");                              \
  {                                                             \
    static Profiler::Timing* slot =                             \
        Profiler::initialize_timing(key, Profiler::chash(key)); \
    Profiler::postprocess(Profiler::chash(key), slot);          \
  }
#endif
#endif

#ifdef INSTRUMENTATION

#define FADDS(n) Profiler::flop_counter.fadds += (n);
#define FMULS(n) Profiler::flop_counter.fmuls += (n);
#define FDIVS(n) Profiler::flop_counter.fdivs += (n);
#define FSQRT(n) Profiler::flop_counter.fsqrts += (n);

#else

#define FADDS(n)
#define FMULS(n)
#define FDIVS(n)
#define FSQRT(n)

#endif

namespace Profiler {

// rdtsc struct
struct rdtsc_struct {
  double cycles() const { return (uint64_t(hi) << 32) + lo; }
  uint32_t hi;
  uint32_t lo;
};

inline double operator-(rdtsc_struct const& end, rdtsc_struct const& begin) {
  return end.cycles() - begin.cycles();
}

struct FlopCounter {
  unsigned fadds = 0;
  unsigned fmuls = 0;
  unsigned fdivs = 0;
  unsigned fsqrts = 0;

  void reset() {
    fadds = 0;
    fmuls = 0;
    fdivs = 0;
    fsqrts = 0;
  }
};

// Timing statistics
struct Timing {
  Timing* parent;
  size_t key_hash;
  unsigned call_count;
  double total_runtime;
  double min;
  double max;

#ifdef INSTRUMENTATION
  unsigned fadds = 0;
  unsigned fmuls = 0;
  unsigned fdivs = 0;
  unsigned fsqrts = 0;
#endif

  Timing() {
    key_hash = std::numeric_limits<size_t>::max();  // mark unused
    call_count = 0;
    total_runtime = 0;
    min = 9007199254740992;
    max = 0;
  }

  double mean() {
    return total_runtime/call_count;
  }
};

// index of the current start time
extern size_t i;

// start times
extern rdtsc_struct start_times[PROFILER_MAX_DEPTH];

extern rdtsc_struct start_time;
extern rdtsc_struct stop_time;

extern Timing* last_timing;

extern double last_duration;

// timings
extern Timing timings[256][PROFILER_TIMING_BIN_SIZE];

extern Timing* timing_stack[PROFILER_MAX_DEPTH];

#ifdef INSTRUMENTATION
extern FlopCounter flop_counters[PROFILER_MAX_DEPTH];
extern FlopCounter flop_counter;
#endif

extern char timer_keys[PROFILER_MAX_SLOTS][PROFILER_MAX_KEY_LENGTH];

extern size_t timer_keys_top;

// const expr hash
template <size_t N, size_t I = 0>
struct hash_calc {
  static constexpr size_t apply(const char (&s)[N]) {
    return (hash_calc<N, I + 1>::apply(s) ^ s[I]) * 16777619u;
  };
};

template <size_t N>
struct hash_calc<N, N> {
  static constexpr size_t apply(const char (&s)[N]) { return 2166136261u; };
};

template <size_t N>
constexpr size_t chash(const char (&s)[N]) {
  return hash_calc<N>::apply(s);
}

size_t hash(const char* s);

inline void reset() {
    i = 0;
    last_timing = nullptr;
    last_duration = 0;
    // invalidate all timings
    for (size_t i=0; i<256; ++i) {
        for (size_t j=0; j<PROFILER_TIMING_BIN_SIZE; ++j) {
            timings[i][j] = Timing();
        }
    }
    timer_keys_top = 0;
}

// stack containing the keys (only used if in debug mode)
#ifndef NDEBUG
static std::stack<size_t> key_stack;
#endif

inline void preprocess(const size_t key_hash) {
  assert(i >= 0);
  start_times[i] = start_time;  // save previous timing
#ifdef INSTRUMENTATION
  flop_counters[i] = flop_counter;
#endif
  i++;

#ifdef INSTRUMENTATION
  flop_counter.reset();  // just to be sure
#ifdef NDEBUG
  assert(flop_counter.fadds == 0);
  assert(flop_counter.fmuls == 0);
  assert(flop_counter.fdivs == 0);
  assert(flop_counter.fsqrts == 0);
#endif
#endif

// debug
#ifndef NDEBUG
  key_stack.push(key_hash);  // push current key_hash onto a stack
#endif
}

inline Timing* get_last_timing() {
  return last_timing;
}

inline double get_last_duration() {
  return last_duration;
}

inline void postprocess(const size_t key_hash, Timing* timing) {
#ifdef INSTRUMENTATION
  timing->fadds += flop_counter.fadds;
  timing->fmuls += flop_counter.fmuls;
  timing->fdivs += flop_counter.fdivs;
  timing->fsqrts += flop_counter.fsqrts;
  flop_counter.reset();
#endif

  // timing statistics
  // if (timing->call_count!=0) {
  last_duration = stop_time - start_time;
  timing->total_runtime += last_duration;
  timing->min = std::min(timing->min, last_duration);
  timing->max = std::max(timing->max, last_duration);
  //}
  timing->call_count += 1;
  last_timing = timing;
  --i;

  start_time = start_times[i];

#ifdef INSTRUMENTATION
  flop_counter = flop_counters[i];
#endif

// debug
#ifndef NDEBUG
  assert(i >= 0);
  assert(key_stack.top() == key_hash);
  key_stack.pop();
#endif
}

void _initialize();

inline void initialize() {
#ifndef NDEBUG
  std::cerr << "W A R N I N G:\n_____________\n\nprofiler has been built in "
               "debug mode\n\n"
            << std::endl;
#endif
  _initialize();
}

Timing* __attribute__((noinline))
initialize_timing(const char* key, const size_t key_hash);

// void print_data();
}

#endif
