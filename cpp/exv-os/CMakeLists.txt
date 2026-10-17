cmake_minimum_required(VERSION 3.15)
project(ExvOs CXX)

# OS System Integration (`ExvOsThumb.dll` Windows Shell IThumbnailProvider + cross-platform
# thumbnail renderer tests).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXV_CAPI_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../docs" CACHE PATH "dir with exv-capi.h")

if(WIN32)
    add_library(ExvOsThumb SHARED
        src/win_thumb_provider.cpp
        src/exv_os_win.def
    )
    target_include_directories(ExvOsThumb PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${EXV_CAPI_INCLUDE}"
    )
endif()

enable_testing()
add_executable(os_thumb_test tests/os_thumb_test.cpp)
target_include_directories(os_thumb_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${EXV_CAPI_INCLUDE}"
)
add_test(NAME os_thumb_test COMMAND os_thumb_test)
