cmake_minimum_required(VERSION 3.0)

add_compile_options(-std=c++11)

add_library(profiler STATIC ${CMAKE_CURRENT_SOURCE_DIR}/profiler.cpp)
#target_link_libraries(profiler pcm)

add_library(profiler_instr STATIC ${CMAKE_CURRENT_SOURCE_DIR}/profiler.cpp)
#target_link_libraries(profiler pcm)
target_compile_definitions(profiler_instr PRIVATE -DINSTRUMENTATION)

add_executable(test_bench ${CMAKE_CURRENT_SOURCE_DIR}/test_bench.cpp)
target_link_libraries(test_bench profiler)
target_link_libraries(test_bench pcre)

#target_compile_options(test_bench PUBLIC -S -fverbose-asm)