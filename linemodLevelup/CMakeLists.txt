cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(patch_linemod)

option(USE_CUDA "use cuda or not" OFF)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_FLAGS "-fPIC -fopenmp -Wall -Wno-sign-compare $ENV{CXXFLAGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(USE_CUDA)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0)
add_definitions(-DCUDA_ON)
endif()


# opencv
find_package(OpenCV 3 REQUIRED)
list(APPEND patch_linemod_inc ${OpenCV_INCLUDE_DIRS})
list(APPEND patch_linemod_lib ${OpenCV_LIBS})

# open3d
find_package(Open3D REQUIRED PATHS "$ENV{HOME}/Open3D/build/CMakeFiles")
if (Open3D_FOUND)
    message(STATUS "Found Open3D ${Open3D_VERSION}")

    include_directories(${Open3D_INCLUDE_DIRS})
    link_directories(${Open3D_LIBRARY_DIRS})

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")
else ()
    message(SEND_ERROR "Open3D not found")
endif ()
list(APPEND patch_linemod_lib ${Open3D_LIBRARIES})


# pybind11
#find_package(PythonLibs 3 REQUIRED)
find_package(pybind11 REQUIRED)
include(Cmake/FindNumPy.cmake)
include_directories(${INCLUDE_DIRECTORIES}
    ${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})


add_subdirectory(cuda_renderer)
add_subdirectory(cuda_icp)
list(APPEND patch_linemod_lib cuda_renderer cuda_icp)


set(SOURCE_PYBIND
    pybind11.cpp
    np2mat/ndarray_converter.cpp)

set(patch_linemod_srcs
    linemodLevelup.cpp pose_refine.cpp)


add_library(${PROJECT_NAME} ${patch_linemod_srcs})
target_include_directories(${PROJECT_NAME} PUBLIC ${patch_linemod_inc})
target_link_libraries(${PROJECT_NAME} PUBLIC ${patch_linemod_lib})

pybind11_add_module(${PROJECT_NAME}_pybind ${SOURCE_PYBIND})
target_link_libraries(${PROJECT_NAME}_pybind PRIVATE patch_linemod)

add_executable(${PROJECT_NAME}_test test.cpp)
target_link_libraries(${PROJECT_NAME}_test patch_linemod)

