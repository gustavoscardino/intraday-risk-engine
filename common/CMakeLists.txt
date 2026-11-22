# Code shared by the three engines: the common model interface, the RiskMetrics baseline and the
# jump filter. Each engine adds this folder to its own build:
#   add_subdirectory(<path to common> ${CMAKE_BINARY_DIR}/riskcommon)
#   target_link_libraries(<engine library> PUBLIC riskcommon)
cmake_minimum_required(VERSION 3.16)
if(NOT TARGET riskcommon)
  add_library(riskcommon STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/riskmetrics.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/jump_filter.cpp)
  target_include_directories(riskcommon PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
  target_compile_features(riskcommon PUBLIC cxx_std_17)
  if(MSVC)
    target_compile_options(riskcommon PRIVATE /W4 /permissive-)
  else()
    target_compile_options(riskcommon PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
  endif()
  # unit tests, registered in the ctest of the engine that builds this folder
  add_executable(riskcommon_tests ${CMAKE_CURRENT_LIST_DIR}/tests/test_common.cpp)
  target_link_libraries(riskcommon_tests PRIVATE riskcommon)
  if(MINGW)
    target_link_libraries(riskcommon_tests PRIVATE -static)
  endif()
  enable_testing()
  add_test(NAME riskcommon_unit_tests COMMAND riskcommon_tests)
endif()
