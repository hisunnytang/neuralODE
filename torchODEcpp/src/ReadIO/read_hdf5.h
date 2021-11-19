#ifndef READ_HDF5_H_
#define READ_HDF5_H_

#include <string>
#include "H5Cpp.h"
#include <iostream>

int readdata_from_hdf5( std::string filename, int batch_size, unsigned long idx_start, double *output_buffer);

using namespace H5;
// https://support.hdfgroup.org/HDF5/doc/cpplus_RM/readdata_8cpp-example.html
// https://www.scivision.dev/hdf5-with-cmake/
//
//

class HDF5_DataLoader{

    int bsz;
    std::string filename;
    H5File    *file;
    DataSet   dataset;
    DataSpace dataspace;
    DataSpace memspace;
    DataSpace memspace2;

    public:   
        HDF5_DataLoader(std:: string filename, int batch_size);
        int SlicingIC  (int batch_size, unsigned long idx_start);
        int SlicingPath(int batch_size, unsigned long idx_start);
        int GetBatch(int batch_size, unsigned long idx_start);
        double *initial_condition;
        double *time_axis;
        double *paths;
        ~HDF5_DataLoader(){delete file; free(initial_condition); free(paths); };

};
#endif // READ_HDF5_H_
