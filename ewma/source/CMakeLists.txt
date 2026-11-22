cmake_minimum_required(VERSION 3.16)
project(ewmar LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(ewmar STATIC
  src/math.cpp
  src/market_data.cpp
  src/seasonality.cpp
  src/ewma.cpp
  src/risk.cpp
  src/models.cpp
  src/engine.cpp
  src/data_quality.cpp
  src/backtest.cpp
  src/replay.cpp)
target_include_directories(ewmar PUBLIC include)
# the code shared by the three engines (model interface, RiskMetrics baseline, jump filter)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../common ${CMAKE_BINARY_DIR}/riskcommon)
target_link_libraries(ewmar PUBLIC riskcommon)
if(MSVC)
  target_compile_options(ewmar PRIVATE /W4 /permissive-)
  target_compile_definitions(ewmar PUBLIC _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(ewmar PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
endif()

if(MINGW)
  # self-contained programs: no libstdc++ / winpthread DLLs needed at run time
  set(EWMAR_STATIC_RUNTIME -static)
  # UTF-8 manifest (paths with accents), linked into each program as a resource object. WinLibs
  # GCC does not link a default manifest of its own, so the program carries exactly this one.
  # windres runs as a plain command: its own include options break on folders with spaces.
  enable_language(RC)
  set(EWMAR_MANIFEST_OBJ "${CMAKE_CURRENT_BINARY_DIR}/windows_utf8_manifest.o")
  add_custom_command(
    OUTPUT "${EWMAR_MANIFEST_OBJ}"
    COMMAND ${CMAKE_RC_COMPILER} "${CMAKE_CURRENT_SOURCE_DIR}/apps/windows_utf8.rc" -O coff -o "${EWMAR_MANIFEST_OBJ}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/apps/windows_utf8.rc"
    VERBATIM)
  set_source_files_properties("${EWMAR_MANIFEST_OBJ}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  set(EWMAR_MANIFEST_SOURCE "${EWMAR_MANIFEST_OBJ}")
elseif(MSVC)
  set(EWMAR_MANIFEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/apps/windows_utf8.manifest")
endif()

function(ewmar_app name src)
  add_executable(${name} ${src} ${EWMAR_MANIFEST_SOURCE})
  target_link_libraries(${name} PRIVATE ewmar ${EWMAR_STATIC_RUNTIME})
  if(MSVC)
    target_compile_options(${name} PRIVATE /W4)
  else()
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
  endif()
endfunction()

ewmar_app(ewma_risk apps/ewma_risk.cpp)
ewmar_app(benchmark apps/benchmark.cpp)
ewmar_app(ewmar_tests tests/test_main.cpp)

# programs copied to ewma/bin, where the README and the run scripts expect them
set(EWMAR_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../bin" CACHE PATH "Folder that receives the built programs")
foreach(app ewma_risk benchmark)
  add_custom_command(TARGET ${app} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${EWMAR_BIN_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${app}>" "${EWMAR_BIN_DIR}")
endforeach()

# datasets (data/, next to ewma) for the end-to-end tests
set(EWMAR_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../data" CACHE PATH "Dataset folder")

enable_testing()
add_test(NAME unit_tests COMMAND ewmar_tests)
if(EXISTS "${EWMAR_DATA_DIR}/synthetic/prices_15m.csv")
  add_test(NAME replay_clean COMMAND ewma_risk --data "${EWMAR_DATA_DIR}/synthetic" --es-sims 200
           --out ${CMAKE_BINARY_DIR}/test_clean)
  add_test(NAME replay_degraded COMMAND ewma_risk --data "${EWMAR_DATA_DIR}/synthetic" --es-sims 200
           --prices "${EWMAR_DATA_DIR}/synthetic/prices_15m_degraded.csv" --out ${CMAKE_BINARY_DIR}/test_degraded)
  add_test(NAME replay_degraded_nodq COMMAND ewma_risk --data "${EWMAR_DATA_DIR}/synthetic" --es-sims 200 --no-dq
           --prices "${EWMAR_DATA_DIR}/synthetic/prices_15m_degraded.csv" --out ${CMAKE_BINARY_DIR}/test_degraded_nodq)
endif()
if(EXISTS "${EWMAR_DATA_DIR}/b3_real/prices_15m.csv")
  add_test(NAME replay_real_b3 COMMAND ewma_risk --data "${EWMAR_DATA_DIR}/b3_real" --es-sims 200
           --out ${CMAKE_BINARY_DIR}/test_b3)
endif()
