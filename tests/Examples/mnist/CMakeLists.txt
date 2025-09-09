cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mnist_openfhe)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wno-error=sign-compare")

find_package(OpenFHE CONFIG REQUIRED)
if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set( CMAKE_CXX_FLAGS "${OpenFHE_CXX_FLAGS} -Werror -Wno-error=sign-compare")
add_link_options(-Wl,--no-as-needed)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)


link_directories( ${OpenFHE_LIBDIR} )
link_directories( ${OPENMP_LIBRARIES} )
link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/pre-built-library )

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_RPATH "${OpenFHE_LIBDIR}")
set(CMAKE_INSTALL_RPATH "${OpenFHE_LIBDIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# If a symbol lookup error occurs, you need to set the LD_LIBRARY_PATH environment to the OpenFHE lib directory.
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/OpenFHE/lib

if(BUILD_STATIC)
    set( CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries( ${OpenFHE_STATIC_LIBRARIES} )
    message(STATUS "Build with static libs")
else()
    set( CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS} )
    link_libraries( ${OpenFHE_SHARED_LIBRARIES} )
endif()

file(GLOB SOURCES
    mnist_openfhe.h
    mnist_openfhe.cpp
    mnist_openfhe_main.cpp
)



add_executable(mnist_openfhe ${SOURCES})
target_link_libraries(mnist_openfhe "${TORCH_LIBRARIES}")

set_property(TARGET mnist_openfhe PROPERTY CXX_STANDARD 17)

