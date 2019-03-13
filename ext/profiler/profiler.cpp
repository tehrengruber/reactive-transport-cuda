#include "profiler.hpp"

namespace Profiler {

/*
 * State variables
 */
// index of the current start time in the timer stack
size_t i = 0;

// start times
//  whenever a new timer is started the (running) timer is pushed here
rdtsc_struct start_times[PROFILER_MAX_DEPTH];

rdtsc_struct start_time; // start time of the current timing
rdtsc_struct stop_time; // stop time of the current timing

Timing* last_timing = nullptr; // pointer to the last timing

double last_duration = 0; // the duration of the last timing

Timing timings[256][PROFILER_TIMING_BIN_SIZE]; // timings

// array containing the keys of all timers
char timer_keys[PROFILER_MAX_SLOTS][PROFILER_MAX_KEY_LENGTH];

// next free slot in the timer_keys array
size_t timer_keys_top = 0;

#ifndef NODEBUG
const bool debug = true;
#else
const bool debug = false;
#endif

#ifdef INSTRUMENTATION
FlopCounter flop_counters[PROFILER_MAX_DEPTH];
FlopCounter flop_counter;
#endif

// string hash function
size_t hash(const char* s) {
  size_t h = 2166136261u;
  size_t N = strlen(s);
  for (int i=N; i>=0; --i) {
    h = (h ^ s[i]) * 16777619u;
  }
  return h;
}

void _initialize() {
#ifdef PROFILER_USE_PCM
  /*
   * Initialize
   */
  std::cout << "==============================================================" << std::endl;
  std::cout << "= IntelPCM is about to be initialized" << std::endl;
  std::cout << "==============================================================" << std::endl;
  intel_PCM_instance = PCM::getInstance();
  intel_PCM_instance->resetPMU();
  PCM::ErrorCode status = intel_PCM_instance->program();
  if (status == PCM::Success) {
      std::cout << "==============================================================" << std::endl;
      std::cout << std::endl;
      std::cout << "==============================================================" << std::endl;
      std::cout << "= IntelPCM initialized" << std::endl;
      std::cout << "==============================================================" << std::endl;
      std::cout << "Detected: " << intel_PCM_instance->getCPUBrandString() << std::endl;
      std::cout << "Codename: " << intel_PCM_instance->getUArchCodename() << std::endl;
      std::cout << "Stepping: " << intel_PCM_instance->getCPUStepping() << std::endl;
      std::cout << "==============================================================" << std::endl;
      std::cout << std::endl << std::endl;
  } else {
      std::cout << "Access to Intel(r) Performance Counter Monitor has been denied" << std::endl;
      exit(EXIT_FAILURE);
  }
  // bind to a single core
  //int num_cores = intel_PCM_instance->getNumCores();
  //int core;
  //for (core = num_cores - 1; core >= 0; core -= 1) {
  //    if ( intel_PCM_instance->isCoreOnline(core) ) {
  //        break;
  //    }
  //}
  int core = 1;

  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(core, &cpu_set);
  sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpu_set);
  msr_handle = new MsrHandle(core);
#endif
}

// initialize timing
Timing* __attribute__ ((noinline)) initialize_timing(const char* key, size_t key_hash) {
  assert(timer_keys_top < PROFILER_MAX_SLOTS);
  assert(strlen(key) <= PROFILER_MAX_KEY_LENGTH);
  // save the corresponding key to the hash
  char *s = timer_keys[timer_keys_top];
  const char *s2 = key;
  size_t n = PROFILER_MAX_KEY_LENGTH;
  while (*s2 != '\0') {
  	*s++ = *s2++;
  	--n;
  }
  //strncpy(timer_keys[timer_keys_top], key, PROFILER_MAX_KEY_LENGTH);
  timer_keys_top++;
  // assign a slot to the timing
  Timing* timing = timings[key_hash % 255]; // first bin
	while (timing->key_hash != std::numeric_limits<size_t>::max()) {
      // TODO: warning
        if (timing->key_hash == key_hash)
            break;
      //assert(timing->key_hash != key_hash);
      ++timing;
	}
  // occupy slot
  timing->key_hash = key_hash;
  // return a pointer to the slot
  return timing;
}

//void print_data() {
//  std::map<size_t, std::string> timer_keys_parsed;
//  for(int i=0; i<ki; ++i) {
//    std::string s = std::string(timer_keys[i]);
//    timer_keys_parsed[hash(s.c_str())] = s;
//  }
//  for (int i=0; i<256; ++i) {
//    Timing* timing = timings[i];
//    while (timing->key_hash != -1) {
//      std::cout << timer_keys_parsed[timing->key_hash] << ": "
//                << timing->total_runtime/timing->call_count << std::endl;
//      ++timing;
//   }
//  }
//}

} // end ns
