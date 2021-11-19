cmake_minimum_required(VERSION 3.5)



# find required packages
find_package(Torch REQUIRED)

#add_subdirectory("../ReadIO")

# add compile flags

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_library(encoderDecoder SHARED torch_EncoderDecoder.cpp)
target_link_libraries(encoderDecoder read_hdf5 "${TORCH_LIBRARIES}")

#target_include_directories(encoderDecoder PUBLIC /home/kwoksun2/data/anaconda3/include)
set_property(TARGET encoderDecoder PROPERTY CXX_STANDARD 14)
