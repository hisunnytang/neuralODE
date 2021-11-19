cmake_minimum_required(VERSION 3.5)

project(sample)

find_package(OpenMP REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "/home/kwoksun2/anaconda3/envs/myenv/lib/python3.9/site-packages/torch/share/cmake")


find_library(HDF5_LIBRARY hdf5 HINTS /home/kwoksun2/anaconda3/lib)
find_library(HDF5_HL_LIBRARY hdf5_hl HINTS /home/kwoksun2/anaconda3/lib)
#find_library(OMP_LIBRARY     gomp HINTS /home/kwoksun2/anaconda3/lib)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_subdirectory(src/ReadIO)
add_subdirectory(src/EncoderDecoder)
add_subdirectory(src/CvodeSolver)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

set(NUM_THREADS 1 )

add_executable(sample sample.cpp)
target_link_libraries(sample PUBLIC encoderDecoder latent_solver ${HDF5_LIBRARY} ${HDF5_HL_LIBRARY} ${TORCH_LIBRARIES} )

if(OpenMP_CXX_FOUND)
    target_link_libraries(sample PUBLIC OpenMP::OpenMP_CXX)
endif()

#target_compile_definitions(sample PRIVATE NUM_THREADS=${NUM_THREADS})
#execute_process(COMMAND ./sample)
#find_package(OpenMP)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(sample PUBLIC OpenMP::OpenMP_CXX)
#endif()
