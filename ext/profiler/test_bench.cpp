#include <cmath>
#include "profiler.hpp"
#include "profiler_io.cpp"

#define REPITITIONS 10000

double v1[20];
double v2[20];
double v2_inv[20];

double trash_can = 0;

#include <iostream>
#include <fstream>
#include <string>
#include <stdint.h>
#include <pcre.h>

using namespace std;


uint32_t cpufreq()
{
    //uint32_t cpuFreq = 0;
    //// CPU frequency is stored in /proc/cpuinfo in lines beginning with "cpu MHz"
    //string pattern = "^cpu MHz\\s*:\\s*(\\d+)";
    //const char* pcreErrorStr = NULL;
    //int pcreErrorOffset = 0;
    //pcre* reCompiled = pcre_compile(pattern.c_str(), PCRE_ANCHORED, &pcreErrorStr,
    //    &pcreErrorOffset, NULL);
    //if (reCompiled == NULL) {
    //    return 0;
    //}
    //ifstream ifs("/proc/cpuinfo");
    //if (ifs.is_open()) {
    //    string line;
    //    int results[10];
    //    while (ifs.good()) {
    //        getline(ifs, line);
    //        int rc = pcre_exec(reCompiled, 0, line.c_str(), line.length(),
    //            0, 0, results, 10);
    //        if (rc < 0)
    //            continue;
    //        // Match found - extract frequency
    //        const char* matchStr = NULL;
    //        pcre_get_substring(line.c_str(), results, rc, 1, &(matchStr));
    //        cpuFreq = atol(matchStr);
    //        pcre_free_substring(matchStr);
    //        break;
    //    }
    //}
    //ifs.close();
    //pcre_free(reCompiled);
    //return cpuFreq;
    return 1800;
}

void setup()
{
  for (unsigned i=0; i<20;++i) {
    v1[i] = 2;
    v2[i] = rand();
    v2_inv[i] = 1./v2[i];
  }
}

__attribute__((optimize("no-tree-vectorize"))) int main() {
  Profiler::initialize();
  volatile int a=0;
  for (unsigned i=0; i<100000; ++i)
    a+=1;

  /*
   * Timing initialization cost
   */
  PROFILER_START("initialization_cost");
  PROFILER_START("initialization_cost.inner");
  PROFILER_STOP("initialization_cost.inner");
  PROFILER_STOP("initialization_cost");
  std::cout << "initialization cost: " << Profiler::get_last_duration() << std::endl;
  /*
   * Cost to do a timing
   */
 for (unsigned i=0; i<1000; ++i) {
   PROFILER_START("timing_cost");
   PROFILER_START("timing_cost.inner");
   PROFILER_STOP("timing_cost.inner");
   PROFILER_STOP("timing_cost");
  }
  std::cout << "timing cost: " << std::endl
            << " min: " << Profiler::get_last_timing()->min << std::endl
            << " mean: " << Profiler::get_last_timing()->mean() << std::endl;

  /*
   * Determine rdtsc clocks per real clocks
   */
  PROFILER_START("sleep");
  sleep(5);
  PROFILER_STOP("sleep");
  double rdtsc_clocks_per_real_clocks = Profiler::get_last_duration()/cpufreq()/1e6/5;
  std::cout << "frequency: " << cpufreq() << std::endl;
  std::cout << "rdtsc clocks/real clocks 1: " << rdtsc_clocks_per_real_clocks << std::endl;

  /*
   * Determine inner timing overhead (time between start and stop without any instructions inbetween)
   */
  for (unsigned i=0; i<REPITITIONS; ++i) {
    PROFILER_START("timing_overhead");
    PROFILER_STOP("timing_overhead");
  }
  double timing_overhead = Profiler::get_last_timing()->min;
  std::cout << "timing overhead: " << std::endl
            << " min: " << Profiler::get_last_timing()->min << std::endl
            << " mean: " << Profiler::get_last_timing()->mean() << std::endl;

  /*
   * Do something that we now it takes 100 cycles
   */
  // muls
  setup();
  for (unsigned i=0; i < REPITITIONS; ++i) {
    PROFILER_START("mul");
    double acc0 = v1[0];
    double acc1 = v1[1];
    double acc2 = v1[2];
    double acc3 = v1[3];
    double acc4 = v1[4];
    double acc5 = v1[5];
    double acc6 = v1[6];
    double acc7 = v1[7];
    double acc8 = v1[8];
    double acc9 = v1[9];
    for (unsigned i=0; i < 10; ++i) {
     // timings change only slightly after 10 accumulators, but are still
     //  performing better at first.
     acc0  *= v2[0];
     acc1  *= v2[1];
     acc2  *= v2[2];
     acc3  *= v2[3];
     acc4  *= v2[4];
     acc5  *= v2[5];
     acc6  *= v2[6];
     acc7  *= v2[7];
     acc8  *= v2[8];
     acc9  *= v2[9];
     acc0  *= v2_inv[0];
     acc1  *= v2_inv[1];
     acc2  *= v2_inv[2];
     acc3  *= v2_inv[3];
     acc4  *= v2_inv[4];
     acc5  *= v2_inv[5];
     acc6  *= v2_inv[6];
     acc7  *= v2_inv[7];
     acc8  *= v2_inv[8];
     acc9  *= v2_inv[9];
    }
    PROFILER_STOP("mul");
    trash_can += acc0+acc1+acc2+acc3+acc4+acc5+acc6+acc7+acc8+acc9;
  }
  double timing_100_cycles_muls = Profiler::get_last_timing()->min;
  std::cout << "200 muls take (expected ~100): " << std::endl
            << " min: " << (timing_100_cycles_muls-timing_overhead)/rdtsc_clocks_per_real_clocks << std::endl
            << " mean: " << (Profiler::get_last_timing()->mean()-timing_overhead)/rdtsc_clocks_per_real_clocks << std::endl;

  // nops
  for (unsigned i=0; i < REPITITIONS; ++i) {
    PROFILER_START("nop");
    asm volatile("nop;nop;nop;nop;"); // 1 cycle
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;"); // 1 cycle
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;"); // 1 cycle
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;"); // 1 cycle
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;"); // 1 cycle
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;"); // 1 cycle
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;"); // 1 cycle
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;"); // 1 cycle
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;"); // 1 cycle
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;"); // 1 cycle
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    asm volatile("nop;nop;nop;nop;");
    PROFILER_STOP("nop");
  }
  double timing_100_cycles_nops = Profiler::get_last_timing()->min;
  std::cout << "400 nops take (expected ~100): " << std::endl
            << " min:" << (timing_100_cycles_nops-timing_overhead)/rdtsc_clocks_per_real_clocks << std::endl
            << " mean:" << (Profiler::get_last_timing()->mean()-timing_overhead)/rdtsc_clocks_per_real_clocks << std::endl;

  Profiler::IO::print_timings();
  Profiler::IO::json_export("nop_timings.json");
  std::cout << std::endl << trash_can << std::endl;
}
