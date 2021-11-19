cmake_minimum_required(VERSION 3.5)

# find required packages
find_library(HDF5_LIBRARY hdf5 HINTS /home/kwoksun2/anaconda3/lib)
find_library(HDF5_CPP_LIBRARY hdf5_cpp HINTS /home/kwoksun2/anaconda3/lib)

# add compile flags
add_library(read_hdf5 read_hdf5.cpp)

target_link_libraries(read_hdf5  ${HDF5_LIBRARY} ${HDF5_CPP_LIBRARY} )
target_include_directories(read_hdf5 PUBLIC /home/kwoksun2/anaconda3/include)
set_property(TARGET read_hdf5 PROPERTY CXX_STANDARD 14)
