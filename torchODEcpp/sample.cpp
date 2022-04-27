#include "sample.h"
#include <chrono>
#include "hdf5.h"
#include "hdf5_hl.h"

using namespace std::chrono;

class DengoFieldData{
    public:
        int    batch_size;
        int    nthreads;
        int    nchem;
        double *H2I_Density;
        double *H2II_Density;
        double *HI_Density;
        double *HII_Density;
        double *HM_Density;
        double *HeI_Density;
        double *HeII_Density;
        double *HeIII_Density;
        double *de_Density;
        double *ge_Density;
        double *density;
        double *input_data;

        DengoFieldData(std::string filename, int batch_size, int nchem, int nthreads );
        ~DengoFieldData();
        int LoadInputData(unsigned long start_idx, int tid);
};

int DengoFieldData::LoadInputData(unsigned long start_idx, int tid){
    
    // number of cells per batch as initial conditions
    unsigned long ncells = batch_size * nchem * tid;

    unsigned long d;
    double *tid_input = &input_data[ncells];
    int j = 0;

    //printf("tid  = %d; start_idx = %lu; tid_idx = %lu\n", tid, start_idx, ncells);

    for ( d = start_idx; d < start_idx + batch_size; d++  ){
        tid_input[j]  = H2I_Density[d] ;
        j++;
        tid_input[j]  = H2II_Density[d] ;
        j++;
        tid_input[j]  = HI_Density[d] ;
        j++;
        tid_input[j]  = HII_Density[d] ;
        j++;
        tid_input[j]  = HM_Density[d] ;
        j++;
        tid_input[j]  = HeI_Density[d] ;
        j++;
        tid_input[j]  = HeII_Density[d] ;
        j++;
        tid_input[j]  = HeIII_Density[d] ;
        j++;
        tid_input[j]  = de_Density[d] ;
        j++;
        tid_input[j]  = ge_Density[d] ;
        //printf("tid_input[%d] = %0.5g\n", j, tid_input[j]);
        j++;
    }
    return 0;
}

DengoFieldData::~DengoFieldData(){
    free(H2I_Density);
    free(H2II_Density);
    free(HI_Density);
    free(HII_Density);
    free(HM_Density);
    free(HeI_Density);
    free(HeII_Density);
    free(HeIII_Density);
    free(de_Density);
    free(ge_Density);
    free(density);
    free(input_data);
}

// using field as shared data
DengoFieldData::DengoFieldData(std::string filename, int bsz, int nspecies, int nthrds )
: batch_size(bsz), nchem(nspecies), nthreads(nthrds)
{

    const char *fileCstring = filename.c_str();


    hid_t file_id;
    hsize_t dims[1];
    file_id = H5Fopen(fileCstring, H5F_ACC_RDWR, H5P_DEFAULT);

    H5LTget_dataset_info(file_id, "/ge", dims, NULL, NULL);
    printf("ncells = %lu\n", dims[0]);
    printf("file in process: %s\n", fileCstring);

    unsigned long total_memspace = dims[0];
    H2I_Density  = (double *) malloc(sizeof(double)* total_memspace);
    H2II_Density = (double *) malloc(sizeof(double)* total_memspace);
    HI_Density  = (double *) malloc(sizeof(double)* total_memspace);
    HII_Density = (double *) malloc(sizeof(double)* total_memspace);
    HM_Density  = (double *) malloc(sizeof(double)* total_memspace);
    HeI_Density   = (double *) malloc(sizeof(double)* total_memspace);
    HeII_Density  = (double *) malloc(sizeof(double)* total_memspace);
    HeIII_Density = (double *) malloc(sizeof(double)* total_memspace);
    de_Density = (double *) malloc(sizeof(double)* total_memspace);
    ge_Density = (double *) malloc(sizeof(double)* total_memspace);
    density    = (double *) malloc(sizeof(double)* total_memspace);

    H5LTread_dataset_double(file_id, "/H2I",  &H2I_Density[0]);
    H5LTread_dataset_double(file_id, "/H2II", &H2II_Density[0]);
    H5LTread_dataset_double(file_id, "/HI", &HI_Density[0]);
    H5LTread_dataset_double(file_id, "/HII", &HII_Density[0]);
    H5LTread_dataset_double(file_id, "/HM", &HM_Density[0]);
    H5LTread_dataset_double(file_id, "/HeI", &HeI_Density[0]);
    H5LTread_dataset_double(file_id, "/HeII", &HeII_Density[0]);
    H5LTread_dataset_double(file_id, "/HeIII", &HeIII_Density[0]);
    H5LTread_dataset_double(file_id, "/de", &de_Density[0]);
    H5LTread_dataset_double(file_id, "/ge", &ge_Density[0]);
    H5LTread_dataset_double(file_id, "/density", &density[0]);
    int status = H5Fclose(file_id);
    // initialize a shared memory for thread private data
    
    input_data = (double *) malloc(sizeof(double) * nthreads * batch_size * nchem);

}

int main()
{
    //HelloFunc();

    torch::set_num_threads(40);
    //ParallelReadHDF5();
    std::string filelocation = "/mnt/gv0/homes/kwoksun2/campus_cluster/test-enzo-dengo/cvode_enzo_64_1e-6";
    std::string ic_file = filelocation+"/new_dd0053_chemistry_3.hdf5"; 

    int batch_size = 2048;
    int nchem      = 10;
    int nthreads   = 40;
    int latent_dim = 3;
    
    double start = omp_get_wtime();
    DengoFieldData field_data(ic_file, batch_size, nchem, nthreads);
    double end = omp_get_wtime();
    printf("DataLoader took %f seconds\n", end - start);

    int nloops = 200;

    double T1 = 1e14;
    double reltol = 1e-5;
    int nspecies = 10;

    #pragma omp parallel
    {   
        unsigned long idx_start  = 0;
        unsigned long i, id, nthrds;
        double *ic_ptr;
    
        EncoderDecoder encdec = EncoderDecoder(batch_size, nspecies, latent_dim, T1, model_location);
        user_data      data   = initialize_user_data(model_location, batch_size, latent_dim);
        LatentSolver   s      = LatentSolver(batch_size, latent_dim, reltol, encdec.get_init_latent_ptr(), encdec.get_invtff_ptr(), (void*) &data);
        

        id     = omp_get_thread_num();
        nthrds = omp_get_num_threads();

        for (i=id; i < nloops; i = i+nthrds)
        {
            idx_start = i * batch_size;
            field_data.LoadInputData(idx_start, id);
            ic_ptr = &(field_data.input_data[id*batch_size*nspecies]);
        
            encdec.EncodeToLatent(batch_size, ic_ptr);
            //printf("Encodeee::: getting[%d] = %lu\n", id, idx_start);
            //std::cout << "pointer to latent" << Encoder_Decoder->get_init_latent_ptr()<< std::endl;
            
            //std::cout << "pointer to EncDec" << &tmp2 << &tmp1 << std::endl;
            
            //printf("omp thread %d; idx_start = %lu; i = %d; nthrds = %d\n", id, idx_start, i, nthrds);
            //printf("data IC [%d] = %0.5g\n", id,  ic_ptr[0]);
            //printf("latent  [%d] = %0.5g\n", id,  tmp2.get_init_latent_ptr()[0]);
            // 3. Perform integration using the init_latent_buffer
            // TODO: output_latent_buffer
            s.RunSolver(batch_size, encdec.get_init_latent_ptr(), 1.3742*T1);

            // 4. decode the latent_buffer
            encdec.LatentToReal(batch_size, s.get_latent_ptr());


            //for (int k = 0; k < 10; k++)
            //    printf("out[%d][%d][%d] = %0.5g\n", i, id, k, encdec.get_abund_ptr()[k]);
        }
    }

    double end2 = omp_get_wtime();
    printf("Solver loop took %f seconds; percells %0.5g\n", end2 - end, (end2-end)/ batch_size/ nloops );

    return 0;

}
