#include "read_hdf5.h"

using namespace H5;
// https://support.hdfgroup.org/HDF5/doc/cpplus_RM/readdata_8cpp-example.html
// https://www.scivision.dev/hdf5-with-cmake/
//
//
int HDF5_DataLoader::GetBatch(int batch_size, unsigned long idx_start){
    // To populate the buffer
    // first slice the initial condition
    this->SlicingIC  (batch_size, idx_start);
    //this->SlicingPath(batch_size, idx_start);
    return 0;
}

int HDF5_DataLoader::SlicingIC(int batch_size, unsigned long idx_start){
    /*
     * Define hyperslab in the dataset; implicitly giving strike and
     * block NULL.
     */
    hsize_t  offset[3];   // hyperslab offset in the file
    hsize_t  count[3];    // size of the hyperslab in the file
    offset[0] = idx_start;
    offset[1] = 0;
    offset[2] = 0;
    count[0]  = batch_size;
    count[1]  = 10;
    count[2]  = 1;
    
    this->dataspace.selectHyperslab( H5S_SELECT_SET, count, offset  );

    
    // define the output buffer
    this->dataset.read(this->initial_condition, PredType::NATIVE_DOUBLE, memspace, dataspace);
    return 0;
}

int HDF5_DataLoader::SlicingPath(int batch_size, unsigned long idx_start){
    /*
     * Define hyperslab in the dataset; implicitly giving strike and
     * block NULL.
     */
    hsize_t  offset[3];   // hyperslab offset in the file
    hsize_t  count[3];    // size of the hyperslab in the file
    offset[0] = idx_start;
    offset[1] = 0;
    offset[2] = 0;
    count[0]  = batch_size;
    count[1]  = 10;
    count[2]  = 74;
    dataspace.selectHyperslab( H5S_SELECT_SET, count, offset  );
    
    // define the output buffer
    dataset.read(this->paths, PredType::NATIVE_DOUBLE, memspace2, dataspace);
    return 0;
}


HDF5_DataLoader::HDF5_DataLoader(std::string filename, int batch_size){
    // initialize the files, dataset, dataspace
    const char *fileCstring = filename.c_str();
    file    = new H5File(fileCstring, H5F_ACC_RDONLY);
    dataset = file->openDataSet("data_group");

    /* Get the dataspace of the dataset */
    dataspace = dataset.getSpace();
    /*
     * Define Memory Space hyperslab
     */
    hsize_t dimsm[3];
    dimsm[0] = batch_size;
    dimsm[1] = 10;
    dimsm[2] = 1;
    memspace = DataSpace(3, dimsm);

    hsize_t dimsm2[3];
    dimsm2[0] = batch_size;
    dimsm2[1] = 10;
    dimsm2[2] = 74;
    memspace2 = DataSpace(3, dimsm2);

    // allocate memory for the output buffer
    initial_condition = (double *) malloc(sizeof(double)* batch_size* 10);
    paths             = (double *) malloc(sizeof(double)* batch_size* 10 * 74);
    time_axis         = (double *) malloc(sizeof(double)* 74);
}

/*
int main(){

    std::string filelocation = "/mnt/gv0/homes/kwoksun2/campus_cluster/test-enzo-dengo/cvode_enzo_64_1e-6";
    std::string ic_file = filelocation+"/new_dd0053_chemistry_5.hdf5"; 

    int           batch_size = 1024;
    unsigned long idx_start  = 0;
    
    
    HDF5_DataLoader DataLoader(ic_file, batch_size);

    for (int k = 0; k < 1000; k++){
        idx_start += batch_size;
        DataLoader.GetBatch(batch_size, idx_start);
    }
    
    for (int i = 0; i < 10; i++){
        std::cout << DataLoader.initial_condition[i] << std::endl;
    }
    

    //DataLoader.~HDF5_DataLoader();

    
    
    double output_buffer[batch_size*10];
    idx_start = 0;
    for (int k = 0; k < 1000; k++){
        idx_start += batch_size;
        readdata_from_hdf5(ic_file, batch_size, idx_start, output_buffer);    
        //printf("k = %d", k);
    }
    
    
}
*/

/*
int main(){
    // Filename 
    std::string filelocation = "/mnt/gv0/homes/kwoksun2/campus_cluster/test-enzo-dengo/cvode_enzo_64_1e-6";
    std::string ic_file = filelocation+"/new_dd0053_chemistry_5.hdf5"; 

    int           batch_size = 1024;
    unsigned long idx_start  = 0;

    double output_buffer[batch_size*10*74];

    readdata_from_hdf5(ic_file, batch_size, idx_start, output_buffer);

    printf("in main\n");
    for (int j = 0; j < 10; j++){
        std::cout << output_buffer[j*74] << " ";
        std::cout << std::endl;
    }
    return 0;
}
*/



/**
 * Reading Data from HDF5 file
 *
 * @param filename HDF5 file, expected to contain `data_group` dataset
 * @param batch_size the expected output batch_size
 * @param idx_start  the starting index
 * @param output_buffer pointer to the output vector
 *
 *
 */
int readdata_from_hdf5( std::string filename, int batch_size, unsigned long idx_start, double *output_buffer)
{
    /* Open file */
    //std::string filelocation = "/mnt/gv0/homes/kwoksun2/campus_cluster/test-enzo-dengo/cvode_enzo_64_1e-6";
    //std::string ic_file = filelocation+"/new_dd0053_chemistry_5.hdf5"; 

    // https://stackoverflow.com/questions/51431246/undefined-symbol-for-hdf5-when-using-g
    const char *fileCstring = filename.c_str();

    H5File  file(fileCstring, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet("data_group");

    /* Get the dataspace of the dataset */
    DataSpace dataspace = dataset.getSpace();

    /*
     * Define hyperslab in the dataset; implicitly giving strike and
     * block NULL.
     */

    hsize_t  offset[3];   // hyperslab offset in the file
    hsize_t  count[3];    // size of the hyperslab in the file
    offset[0] = idx_start;
    offset[1] = 0;
    offset[2] = 0;
    count[0]  = batch_size;
    count[1]  = 10;
    count[2]  = 1;
    dataspace.selectHyperslab( H5S_SELECT_SET, count, offset  );


    /*
     * Define Memory Space hyperslab
     */
    hsize_t dimsm[3];
    dimsm[0] = batch_size;
    dimsm[1] = 10;
    dimsm[2] = 1;
    DataSpace memspace(3, dimsm);

    // define the output buffer
    dataset.read(output_buffer, PredType::NATIVE_DOUBLE, memspace, dataspace);
    
    return 0;
    

}
