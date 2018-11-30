cmake_minimum_required(VERSION 3.10)
project(untitled)

# can be launched with: 98, 11, 14 C++ standarts
# it may require to slightly adjust codebase (minor things such as: for range loops, auto type, etc)
# to support older C++ standarts
set(CMAKE_CXX_STANDARD 17)

add_executable(main.cpp
               net.h net.cpp
               neuron.cpp neuron.h
               trainingSet.cpp trainingSet.h)
