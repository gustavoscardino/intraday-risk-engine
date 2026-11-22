cmake_minimum_required(VERSION 3.16)
project(garcht LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GARCHT_NATIVE "Optimize for the build machine (-march=native)" OFF)
option(GARCHT_SANITIZE "Build with address/undefined sanitizers (GCC/Clang, non-Windows)" OFF)

find_package(Threads REQUIRED)

add_library(garcht STATIC
  src/math.cpp
  src/optimize.cpp
  src/garch.cpp
  src/risk.cpp
  src/backtest.cpp
  src/seasonality.cpp
  src/portfolio.cpp
  src/market_data.cpp
  src/data_quality.cpp
  src/revaluation.cpp
  src/synthetic.cpp
  src/replay.cpp)
target_include_directories(garcht PUBLIC include)
target_link_libraries(garcht PUBLIC Threads::Threads)

if(MSVC)
  target_compile_options(garcht PRIVATE /W4 /permissive-)
  # fopen/sscanf/strerror are used portably; silence MSVC's "use the _s variants" advice
  target_compile_definitions(garcht PUBLIC _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(garcht PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
  if(GARCHT_NATIVE)
    target_compile_options(garcht PUBLIC -march=native)
  endif()
  if(GARCHT_SANITIZE)
    target_compile_options(garcht PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(garcht PUBLIC -fsanitize=address,undefined)
  endif()
endif()

if(MINGW)
  # Self-contained executables (no libstdc++/winpthread DLLs needed at run time).
  set(GARCHT_STATIC_RUNTIME -static)
endif()

# Windows manifest with UTF-8 as the process code page (Windows 10 1903+), so that paths with
# accents (C:/Users/José/...) work. GCC always links its own default-manifest.o; ours is built
# under that same name in a folder searched first (-B), so the program carries exactly one.
if(MINGW)
  enable_language(RC)
  set(GARCHT_MANIFEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/winmanifest")
  add_custom_command(
    OUTPUT "${GARCHT_MANIFEST_DIR}/default-manifest.o"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GARCHT_MANIFEST_DIR}"
    COMMAND ${CMAKE_RC_COMPILER} "${CMAKE_CURRENT_SOURCE_DIR}/apps/windows_utf8.rc" -O coff
            -o "${GARCHT_MANIFEST_DIR}/default-manifest.o"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/apps/windows_utf8.rc"
    VERBATIM)
  add_custom_target(garcht_manifest DEPENDS "${GARCHT_MANIFEST_DIR}/default-manifest.o")
elseif(MSVC)
  set(GARCHT_MANIFEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/apps/windows_utf8.manifest")
endif()

function(garcht_app name src)
  add_executable(${name} ${src} ${GARCHT_MANIFEST_SOURCE})
  target_link_libraries(${name} PRIVATE garcht ${GARCHT_STATIC_RUNTIME})
  if(MINGW)
    add_dependencies(${name} garcht_manifest)
    target_link_options(${name} PRIVATE "-B${GARCHT_MANIFEST_DIR}/")
    set_property(TARGET ${name} APPEND PROPERTY LINK_DEPENDS "${GARCHT_MANIFEST_DIR}/default-manifest.o")
  endif()
  if(MSVC)
    target_compile_options(${name} PRIVATE /W4)
  else()
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
  endif()
endfunction()

garcht_app(generate_data apps/generate_data.cpp)
garcht_app(intraday_risk apps/intraday_risk.cpp)
if(WIN32)
  target_link_libraries(intraday_risk PRIVATE ws2_32)  # --live: TCP feed
endif()
garcht_app(fit_series apps/fit_series.cpp)
garcht_app(benchmark apps/benchmark.cpp)
garcht_app(monte_carlo apps/monte_carlo.cpp)
garcht_app(garcht_tests tests/test_main.cpp)

# The programs are also copied to garch_t/bin (next to this source/ folder), where the README
# and the run scripts expect them.
set(GARCHT_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../bin" CACHE PATH "Folder that receives the built programs")
foreach(app generate_data intraday_risk fit_series benchmark monte_carlo)
  add_custom_command(TARGET ${app} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GARCHT_BIN_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${app}>" "${GARCHT_BIN_DIR}")
endforeach()

# Dataset folder (data/, next to garch_t), used by the real-data integration test.
set(GARCHT_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../data" CACHE PATH "Dataset folder")

enable_testing()
add_test(NAME unit_tests COMMAND garcht_tests)
add_test(NAME generate_dataset COMMAND generate_data --out ${CMAKE_BINARY_DIR}/test_data --days 90)
add_test(NAME replay_clean COMMAND intraday_risk --data ${CMAKE_BINARY_DIR}/test_data --no-options
         --out ${CMAKE_BINARY_DIR}/test_output --window-days 40)
add_test(NAME replay_degraded COMMAND intraday_risk --data ${CMAKE_BINARY_DIR}/test_data --no-options
         --prices ${CMAKE_BINARY_DIR}/test_data/prices_15m_degraded.csv
         --out ${CMAKE_BINARY_DIR}/test_output_degraded --window-days 40 --z-clip 6)
add_test(NAME replay_options COMMAND intraday_risk --data ${CMAKE_BINARY_DIR}/test_data
         --options ${CMAKE_BINARY_DIR}/test_data/options.csv --mc 2000 --es-sims 200
         --out ${CMAKE_BINARY_DIR}/test_output_options --window-days 40)
add_test(NAME monte_carlo_quick COMMAND monte_carlo --seeds 2 --days 70 --window 40 --es-sims 100
         --out ${CMAKE_BINARY_DIR}/test_output_mc)
set_tests_properties(generate_dataset PROPERTIES FIXTURES_SETUP dataset)
set_tests_properties(replay_clean replay_degraded replay_options PROPERTIES FIXTURES_REQUIRED dataset)
if(EXISTS "${GARCHT_DATA_DIR}/b3_real/prices_15m.csv")
  add_test(NAME replay_real_b3 COMMAND intraday_risk --data "${GARCHT_DATA_DIR}/b3_real" --window-days 30
           --es-sims 200 --out ${CMAKE_BINARY_DIR}/test_output_b3)
endif()
