cmake_minimum_required(VERSION 3.16)
project(realized_caviar LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# Reuse the established statistical numerics without changing garch_t.
set(GARCH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../garch_t/source")
add_library(rcaviar_statistics STATIC
  ${GARCH_SOURCE}/src/math.cpp ${GARCH_SOURCE}/src/optimize.cpp ${GARCH_SOURCE}/src/garch.cpp ${GARCH_SOURCE}/src/risk.cpp ${GARCH_SOURCE}/src/backtest.cpp ${GARCH_SOURCE}/src/market_data.cpp
  ${GARCH_SOURCE}/src/data_quality.cpp
  ../../ewma/source/src/math.cpp ../../ewma/source/src/ewma.cpp ../../ewma/source/src/seasonality.cpp)
target_include_directories(rcaviar_statistics PUBLIC ${GARCH_SOURCE}/include)
target_include_directories(rcaviar_statistics PUBLIC ../../ewma/source/include)
# the code shared by the three engines (model interface, jump filter)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../common ${CMAKE_BINARY_DIR}/riskcommon)
target_link_libraries(rcaviar_statistics PUBLIC riskcommon)
add_library(rcaviar STATIC src/model.cpp src/data.cpp src/shared_data.cpp src/replay.cpp)
target_include_directories(rcaviar PUBLIC include)
target_link_libraries(rcaviar PUBLIC rcaviar_statistics)
# Windows (MinGW): UTF-8 code page, so paths with accents work (as in ewma and garch_t), and
# self-contained programs. windres runs as a plain command: its include options break on
# folders with spaces; the object is linked into each program.
if(MINGW)
  enable_language(RC)
  set(RCAVIAR_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/windows_utf8_manifest.o")
  add_custom_command(OUTPUT "${RCAVIAR_MANIFEST}"
    COMMAND ${CMAKE_RC_COMPILER} "${CMAKE_CURRENT_SOURCE_DIR}/apps/windows_utf8.rc" -O coff -o "${RCAVIAR_MANIFEST}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/apps/windows_utf8.rc" VERBATIM)
  set_source_files_properties("${RCAVIAR_MANIFEST}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
endif()
add_executable(realized_caviar apps/replay.cpp)
target_link_libraries(realized_caviar PRIVATE rcaviar rcaviar_statistics)
add_executable(rcaviar_tests tests/test_main.cpp)
target_link_libraries(rcaviar_tests PRIVATE rcaviar rcaviar_statistics)
add_executable(rcaviar_fit apps/fit_series.cpp)
target_link_libraries(rcaviar_fit PRIVATE rcaviar)
if(MINGW)
  foreach(app realized_caviar rcaviar_tests rcaviar_fit)
    target_sources(${app} PRIVATE "${RCAVIAR_MANIFEST}")
    target_link_libraries(${app} PRIVATE -static)
  endforeach()
endif()
foreach(target rcaviar_statistics rcaviar realized_caviar rcaviar_tests rcaviar_fit)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
  endif()
endforeach()
option(RCAVIAR_SANITIZE "Address and undefined behavior sanitizers" OFF)
if(RCAVIAR_SANITIZE AND NOT MSVC)
  foreach(target rcaviar_statistics rcaviar realized_caviar rcaviar_tests rcaviar_fit)
    target_compile_options(${target} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PUBLIC -fsanitize=address,undefined)
  endforeach()
endif()
enable_testing()
add_test(NAME unit_tests COMMAND rcaviar_tests)
add_test(NAME synthetic_replay COMMAND realized_caviar --out ${CMAKE_CURRENT_BINARY_DIR}/test_results)

set(SHARED_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../data")
foreach(dataset synthetic b3_real)
  if(EXISTS "${SHARED_DATA_DIR}/${dataset}/prices_15m.csv")
    if(dataset STREQUAL "synthetic")
      set(shared_ticker XPET3)
    else()
      set(shared_ticker PETR4)
    endif()
    add_test(NAME shared_${dataset} COMMAND realized_caviar
      --data "${SHARED_DATA_DIR}/${dataset}" --ticker ${shared_ticker}
      --out "${CMAKE_CURRENT_BINARY_DIR}/test_shared_${dataset}")
  endif()
endforeach()
if(EXISTS "${SHARED_DATA_DIR}/synthetic/prices_15m_degraded.csv")
  add_test(NAME shared_degraded COMMAND realized_caviar
    --data "${SHARED_DATA_DIR}/synthetic" --ticker XPET3
    --prices "${SHARED_DATA_DIR}/synthetic/prices_15m_degraded.csv"
    --out "${CMAKE_CURRENT_BINARY_DIR}/test_shared_degraded")
endif()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME output_contract COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_outputs.py $<TARGET_FILE:realized_caviar> ${SHARED_DATA_DIR})
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import scipy.optimize"
                  RESULT_VARIABLE RCAVIAR_SCIPY_STATUS OUTPUT_QUIET ERROR_QUIET)
  if(RCAVIAR_SCIPY_STATUS EQUAL 0)
    add_test(NAME scipy_fit_reference COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference/crosscheck_scipy.py $<TARGET_FILE:rcaviar_fit>)
  endif()
endif()
