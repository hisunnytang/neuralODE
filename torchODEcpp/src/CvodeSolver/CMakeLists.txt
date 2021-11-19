cmake_minimum_required(VERSION 3.5)

project("test_latent")

set(CMAKE_PREFIX_PATH "/home/kwoksun2/anaconda3/envs/myenv/lib/python3.9/site-packages/torch/share/cmake")

# ------------------------------------------------------------------------------

#find_library(OMP_LIBRARY     gomp HINTS /home/kwoksun2/anaconda3/lib)
# Specify the path to SUNDIALS header files
set(SUNDIALS_INCLUDE_DIR
  /home/kwoksun2/data/cvode-5.3.0_instdir/include
  CACHE PATH "Location of SUNDIALS header files")

# Specify the path to SUNDIALS libraries
set(SUNDIALS_LIBRARY_DIR
  /home/kwoksun2/data/cvode-5.3.0_instdir/lib
  CACHE PATH "Location of SUNDIALS libraries")

# Find the SUNDIALS libraries
find_library(SUNDIALS_SOLVER_LIB
  sundials_cvode ${SUNDIALS_LIBRARY_DIR}
  DOC "CVODE library")

find_library(SUNDIALS_NVEC_LIB
  sundials_nvecserial ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_SERIAL library")

find_library(SUNDIALS_MANYVEC_LIB
  sundials_nvecmanyvector ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_MANYVECTOR library")

# Set additional libraries
set(SUNDIALS_EXTRA_LIBS  -lm /home/kwoksun2/anaconda3/envs/myenv/x86_64-conda-linux-gnu/sysroot/usr/lib/librt.so CACHE STRING "Additional libraries")

# For SUNDIALS module examples the solver library is not needed
if(NOT SUNDIALS_SOLVER_LIB)
  set(SUNDIALS_SOLVER_LIB "")
endif()

# List of SUNDIALS libraries
set(SUNDIALS_LIBRARIES
  -L${SUNDIALS_LIBRARY_DIR}
  ${SUNDIALS_SOLVER_LIB}
  ${SUNDIALS_NVEC_LIB}
  ${SUNDIALS_MANYVEC_LIB}
  ${SUNDIALS_EXTRA_LIBS})

# ------------------------------------------------------------------------------

# KLU include directory and libraries
set(KLU_INCLUDE_DIR
  /home/kwoksun2/data/dengo-merge/suitesparse/include
  CACHE PATH "Location of KLU header files")

set(KLU_LIBRARIES
  /home/kwoksun2/data/dengo-merge/suitesparse/lib/libklu.so;/home/kwoksun2/data/dengo-merge/suitesparse/lib/libamd.so;/home/kwoksun2/data/dengo-merge/suitesparse/lib/libcolamd.so;/home/kwoksun2/data/dengo-merge/suitesparse/lib/libbtf.so;/home/kwoksun2/data/dengo-merge/suitesparse/lib/libsuitesparseconfig.so
  CACHE STRING "KLU libraries")

# Find the SUNDIALS linear solver libraries
find_library(SUNDIALS_SUNLINSOLKLU_LIB
    sundials_sunlinsolklu ${SUNDIALS_LIBRARY_DIR}
    DOC "SUNDIALS KLU linear solver library")
# ------------------------------------------------------------------------------

# find required packages
find_package(Torch REQUIRED)
find_library(HDF5_LIBRARY hdf5 HINTS /home/kwoksun2/data/anaconda3/lib)
find_library(HDF5_CPP_LIBRARY hdf5_cpp HINTS /home/kwoksun2/data/anaconda3/lib)

# add compile flags

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# source file
add_library(latent_solver latent_cvode_solver.cpp)

# libraries to link against
target_link_libraries(latent_solver ${TORCH_LIBRARIES} ${HDF5_LIBRARY} ${HDF5_CPP_LIBRARY} )
target_link_libraries(latent_solver ${SUNDIALS_LIBRARIES})
target_link_libraries(latent_solver ${SUNDIALS_SUNLINSOLKLU_LIB})
target_link_libraries(latent_solver ${KLU_LIBRARIES} ${OMP_LIBRARY})

# directories to include
target_include_directories(latent_solver PUBLIC ${SUNDIALS_INCLUDE_DIR})
target_include_directories(latent_solver PUBLIC ${KLU_INCLUDE_DIR})
target_include_directories(latent_solver PUBLIC /home/kwoksun2/data/anaconda3/include)

set_property(TARGET latent_solver PROPERTY CXX_STANDARD 14)
