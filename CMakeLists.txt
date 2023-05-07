cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(ENABLE_CUDA ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(DEFAULT_BUILD_TYPE "Release")

project(cuPhilox LANGUAGES C CXX CUDA)
set(${PROJECT_NAME}_VERSION 0.0.0)
enable_language(C CXX CUDA)

# Prohibit in-source build
if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(FATAL_ERROR " does not support in-source CMake builds at this time.")
endif (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})

find_package(CUDA REQUIRED)
find_package(CUDAToolkit)

# Find CUDA ARCH VERSION
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET( ${CUDA_ARCH_LIST})
list (GET CUDA_ARCH_LIST 0 ARCH_NUMBER)
SET(CUDA_ARCH sm_${ARCH_NUMBER})
SET(CUDA_COMPUTE compute_${ARCH_NUMBER})
message(--------------------------------------------)
message(CUDA_ARCH = ${CUDA_ARCH})
message(CUDA_COMPUTE = ${CUDA_COMPUTE})
message(--------------------------------------------)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")

set(CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr --compiler-options -fPIC -g -G -Xcompiler -Wall")
set(CUDA_FLAGS "-gencode code=${CUDA_ARCH},arch=${CUDA_COMPUTE} ${CUDA_FLAGS}")
string(REGEX REPLACE "^sm_" "" ARCH_NUMBER "${CUDA_ARCH}")
if ("${CUDA_ARCH}" STREQUAL "sm_${ARCH_NUMBER}")
	set(CMAKE_CUDA_ARCHITECTURES "${ARCH_NUMBER}")
else()
	message(FATAL_ERROR "Unknown CUDA_ARCH: ${CUDA_ARCH}")
endif()
message(STATUS "Using CUDA architecture: ${CUDA_ARCH}")

set(CUDA_FLAGS "-ccbin=${CMAKE_CXX_COMPILER} ${CUDA_FLAGS}")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -lpthread")

set(MAIN_FILE ${PROJECT_SOURCE_DIR}/philox.cu ${PROJECT_SOURCE_DIR}/philox.cuh)

# Declaring the files
set(ALL_SRC ${MAIN_FILE})
foreach(FNAME IN LISTS ALL_SRC)
    message(${FNAME})
endforeach()

add_compile_options(-Wno-unknown-pragmas)

add_subdirectory(pybind11)
pybind11_add_module(cuPhilox ${ALL_SRC})
link_directories(${CUDA_LIBRARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(cuPhilox PRIVATE ${CUDA_LIBRARIES})
set_target_properties(cuPhilox PROPERTIES POSITION_INDEPENDENT_CODE ON)
