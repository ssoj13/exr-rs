cmake_minimum_required(VERSION 3.15)
project(ExvNuke CXX)

# Foundry Nuke NDK FileReader / FileWriter plugins (`exvReader`, `exvWriter`) over `exv_capi`.
# Builds on Windows (`.dll`), Linux (`.so`), and macOS (`.dylib`) when `-DNUKE_NDK_PATH=...` is
# provided; always builds the SDK-independent `nuke_io_test` harness.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXV_CAPI_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../docs" CACHE PATH "dir with exv-capi.h")
set(NUKE_NDK_PATH "" CACHE PATH "Optional path to Foundry Nuke NDK directory (include/DDImage + libDDImage)")

if(NUKE_NDK_PATH AND EXISTS "${NUKE_NDK_PATH}/include/DDImage/FileReader.h")
    find_library(DDIMAGE_LIB NAMES DDImage HINTS "${NUKE_NDK_PATH}" "${NUKE_NDK_PATH}/lib")
    if(DDIMAGE_LIB)
        foreach(plug exvReader exvWriter)
            add_library(${plug} MODULE src/${plug}.cpp)
            target_include_directories(${plug} PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${EXV_CAPI_INCLUDE}"
                "${NUKE_NDK_PATH}/include"
            )
            target_link_libraries(${plug} PRIVATE "${DDIMAGE_LIB}")
            set_target_properties(${plug} PROPERTIES PREFIX "")
            if(APPLE)
                set_target_properties(${plug} PROPERTIES SUFFIX ".dylib")
            endif()
        endforeach()
    endif()
endif()

enable_testing()
add_executable(nuke_io_test tests/nuke_io_test.cpp)
target_include_directories(nuke_io_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${EXV_CAPI_INCLUDE}"
)
add_test(NAME nuke_io_test COMMAND nuke_io_test)
