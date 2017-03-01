project(Roboman-SDM)
cmake_minimum_required(VERSION 2.8.12)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

#MinGW编译的时候，添加-static选项，去除对其他三个标准库的依赖，单独添加-static-libstdc++，只去除对libstdc++标准的依赖
# g++ needs a compiler flag to enable C++11 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-std=c++11 HAS_CXX11_FLAG)
if (HAS_CXX11_FLAG)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	message("..........................................")
else()
    message("WARNING: THERE IS NO CXX11 FLAG...........")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("src/include")
aux_source_directory("src/include" SOURCES_FILE)

#ADD_LIBRARY(SRC ${SOURCES_FILE})

ADD_EXECUTABLE(Train-SDM src/train_model.cpp)
TARGET_LINK_LIBRARIES(Train-SDM ${OpenCV_LIBS})

ADD_EXECUTABLE(Roboman-SDM-TEST src/test_model.cpp)
TARGET_LINK_LIBRARIES(Roboman-SDM-TEST ${OpenCV_LIBS})
