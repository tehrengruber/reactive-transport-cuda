#include <map>
#include <iomanip>
#include <fstream>
#include <cmath>

// boost property tree
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
using namespace boost::property_tree::json_parser;

namespace Profiler {

namespace IO {

using boost::property_tree::ptree;

namespace detail {

inline int get_tree_depth(ptree& pt, int depth=1) {
  if (pt.empty())
    return depth;

  int new_depth = 1;
  for (auto& node : pt) {
      new_depth = std::max(new_depth, get_tree_depth(node.second, depth+1));
  }
  return new_depth;
}

inline int get_max_width(ptree& pt, int width=0) {
  int new_width = width;
  for (auto& node : pt) {
      new_width = std::max<int>(new_width, width+node.first.size());

      if (!node.second.empty()) {
        new_width = std::max<int>(new_width,
                                  get_max_width(node.second, width+2));
      }
  }
  return new_width;
}

inline int ptree_child_count(ptree& pt, int offset=0) {
  int c=offset;
  for (auto& n : pt) ++c;
  return c;
}

} // end ns detail

ptree convert_timing_to_ptree(Profiler::Timing& timing) {
  ptree pt;
  // iterations
  pt.put<long>("call_count", timing.call_count);

  // cycles
  ptree cycles_pt;
  cycles_pt.put<double>("sum", timing.total_runtime);
  cycles_pt.put<double>("min", timing.min);
  cycles_pt.put<double>("max", timing.max);
  cycles_pt.put<double>("mean", timing.total_runtime/timing.call_count);
  pt.put_child("cycles", cycles_pt);

  #ifdef INSTRUMENTATION
  ptree flops_pt;
  flops_pt.put<unsigned>("fadds", timing.fadds);
  flops_pt.put<unsigned>("fmuls", timing.fmuls);
  flops_pt.put<unsigned>("fdivs", timing.fdivs);
  flops_pt.put<unsigned>("fsqrts", timing.fsqrts);
  flops_pt.put<unsigned>("ftotal", timing.fadds + timing.fmuls + timing.fdivs + timing.fsqrts);
  pt.put_child("flops", flops_pt);
  #endif

  return pt;
}

double runtime_in_subtree(ptree& pt) {
  assert(!pt.empty());

  // calculate timings of all child nodes
  double timings_children = 0.;
  for (auto& node : pt) {
    if (node.first != "timing") {
      timings_children += node.second.get<double>("timing.cycles.sum");
    }
  }
  return timings_children;
}

unsigned long call_count_in_subtree(ptree& pt) {
  assert(!pt.empty());

  unsigned long call_count = 0;

  //auto has_timing = pt.get_child_optional("timing");
  //if(has_timing) {
  //  call_count += pt.get<long>("timing.call_count");
  //}

  // calculate timings of all child nodes
  for (auto& node : pt) {
    if (node.first != "timing") {
      call_count += call_count_in_subtree(node.second);
      call_count += node.second.get<double>("timing.call_count");
    }
  }
  return call_count;
}

void complete_timings(ptree& pt) {
  // print timing of the node (if existent)
  auto has_timing = pt.get_child_optional("timing");
  if(has_timing) {
    pt.put<double>("timing.cycles.excl",
      pt.get<double>("timing.cycles.sum")-runtime_in_subtree(pt));
    pt.put<long>("timing.call_count_children", call_count_in_subtree(pt));
  }

  for (auto& node : pt) {
    if (node.first != "timing") {
      complete_timings(node.second);
    }
  }
}

std::string header() {
  std::stringstream header;
  header << std::setw(12) << "call count" << " "
       << std::setw(12) << "excl %" << " "
       << std::setw(12) << "incl %" << " "
       << std::setw(12) << "excl" << " "
       << std::setw(12) << "incl" << " "
       << std::setw(12) << "mean" << " "
       << std::setw(12) << "min"  << " "
       << std::setw(12) << "max";
  #ifdef INSTRUMENTATION
  header << std::setw(12) << "fadds" << " "
         << std::setw(12) << "fmuls" << " "
         << std::setw(12) << "fdivs" << " "
         << std::setw(12) << "fsqrts" << " "
         << std::setw(12) << "ftotal" << " " << std::endl;
  #endif
  return header.str();
}

void print_timing_tree(ptree& pt, double total_runtime, int stack_column_indent) {
  std::cout << std::string(stack_column_indent, ' ')
            << std::setw(12) << pt.get<int>("call_count") << " "
//            << std::setw(12) << pt.get<int>("call_count_children") << " "
            << std::setw(12) << std::round(100*100*pt.get<double>("cycles.excl")/total_runtime)/100 << " "
            << std::setw(12) << std::round(100*100*pt.get<double>("cycles.sum")/total_runtime)/100 << " "
            << std::setw(12) << pt.get<double>("cycles.excl") << " "
            << std::setw(12) << pt.get<double>("cycles.sum") << " "
            << std::setw(12) << pt.get<double>("cycles.mean") << " "
            << std::setw(12) << pt.get<double>("cycles.min")  << " "
            << std::setw(12) << pt.get<double>("cycles.max");
  #ifdef INSTRUMENTATION
  std::cout << " " << std::setw(12) << pt.get<double>("flops.fadds")
            << " " << std::setw(12) << pt.get<double>("flops.fmuls")
            << " " << std::setw(12) << pt.get<double>("flops.fdivs")
            << " " << std::setw(12) << pt.get<double>("flops.fsqrts")
            << " " << std::setw(12) << pt.get<double>("flops.ftotal");
  #endif
}

void print_tree(std::string key, ptree& pt, double total_runtime, int first_column_width, int level = 0) {
  // end line of the parent
  if (level!=0) std::cout << std::endl;

  // print the key
  std::cout << std::string(2*level, ' ') << key << ": ";

  if (pt.empty()) {
    std::cout << pt.data();
  } else {
    // print timing of the node (if existent)
    auto has_timing = pt.get_child_optional("timing");
    if(has_timing) {
      print_timing_tree(pt.get_child("timing"), total_runtime, first_column_width-2*level-key.size());
    }

    // print children
    int pos=0;
    for (auto& node : pt) {
      // print data
      if (node.first != "timing") {
        print_tree(node.first, node.second, total_runtime, first_column_width, level+1);
      }
      pos++;
    }
  }
}

void print_tree(ptree& pt) {
  // calculate total runtime
  double total_runtime = 0.;
  for (auto& node : pt) {
    if (node.first != "timing") {
      assert(node.second.get_child_optional("timing"));
      total_runtime+=node.second.get<double>("timing.cycles.sum");
    }
  }

  // determine max width of the first column (i.e. the column in which the
  //  call stack is visualized)
  int first_column_width = detail::get_max_width(pt);
  // print header
  std::cout << std::string(first_column_width, ' ') << header() << std::endl;
  // print tree
  for (auto& node : pt) {
    print_tree(node.first, node.second, total_runtime, first_column_width, 0);
    std::cout << std::endl;
  }
}

std::vector<std::pair<std::string, long>> virtual_loop_bodies;

void add_virtual_loop_body(std::string key, long repititions) {
  virtual_loop_bodies.push_back(std::make_pair(key, repititions));
}

void process_virtual_loop_body(ptree& pt, std::string key, long repititions) {
  long call_count = pt.get<long>(key+".timing.call_count");
  double sum = pt.get<double>(key+".timing.cycles.sum");
  Profiler::Timing virtual_timing;
  virtual_timing.call_count = repititions;
  virtual_timing.total_runtime = sum/call_count;
  virtual_timing.min = 0;
  virtual_timing.max = 0;
  //auto bla = convert_timing_to_ptree(virtual_timing);
  //std::cout << bla.get<double>("call_count") << std::endl;
  pt.add_child(key + ".body.timing", convert_timing_to_ptree(virtual_timing));
}

std::vector<std::pair<std::string, long>> additional_timing_information;

template <typename T>
void add_timing_information(std::string key, long value) {
  additional_timing_information.emplace_back(key, value);
}

//ptree parsed_timing_data;
//
//std::string current_section;
//
//void section_begin(std::string name) {
//  current_section = name;
//}
//
//void section_end(std::string name) {
//  assert(name == current_section);
//  ptree pt = get_timings();
//  parsed_timing_data.add_child(name, pt);
//  Profiler::reset();
//}

ptree get_timings() {
  ptree pt;
  // create a std::map from the key hash to the key
  std::map<size_t, std::string> key_hash_map;
  for(int i=0; i<timer_keys_top; ++i) {
    std::string s = std::string(Profiler::timer_keys[i]);
    key_hash_map[hash(s.c_str())] = s;
  }
  // iterate over all collected timings and store them
  for (int i=0; i<256; ++i) {
    Timing* timing = timings[i];
    while (timing->key_hash != -1) {
      pt.add_child(key_hash_map[timing->key_hash] + ".timing",
                      convert_timing_to_ptree(*timing));
      ++timing;
    }
  }
  // process virtual loop bodies
  for (std::pair<std::string, long> v : virtual_loop_bodies) {
    process_virtual_loop_body(pt, v.first, v.second);
  }
  // go through the complete property tree and accummulate timings
  complete_timings(pt);
  // add additional timing information
  for (auto& info : additional_timing_information) {
    pt.put<long>(info.first, info.second);
  }

  return pt;
}

void json_export(const std::string file_name) {
	// process data
	ptree pt = get_timings();
	// open output file
	std::ofstream fs;
	fs.open(file_name, std::ios::out | std::ios::trunc);
	write_json(file_name, pt);
	fs.close();
}

void print_timings() {
  ptree timings = get_timings();
  print_tree(timings);
}

} // end ns IO

} // end ns Profiler
