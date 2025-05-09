cmake_minimum_required(VERSION 3.8.2...3.31)
project(TFLITE_DEMO
  VERSION 1.0
  LANGUAGES C CXX
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${SYSROOT_INCDIR})
link_directories(${SYSROOT_LIBDIR})

find_package(PkgConfig)

list(APPEND TARGET_INCS
  ${SYSROOT_INCDIR}/tensorflow
)

include(CheckIncludeFile)

pkg_check_modules(TFLITE tensorflow-lite-prebuilt)
if(NOT TFLITE_FOUND)
  pkg_check_modules(TFLITE REQUIRED tensorflow-lite)
endif()

list(APPEND TARGET_LIBRARIES ${TFLITE_LIBRARIES} QnnTFLiteDelegate m)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter")

add_executable(tflite_demo tflite_demo.c)

target_include_directories(tflite_demo PUBLIC ${TARGET_INCS})
target_link_libraries(tflite_demo PRIVATE ${TARGET_LIBRARIES})
