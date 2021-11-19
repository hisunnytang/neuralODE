#include "torch_EncoderDecoder.h"


EncoderDecoder::EncoderDecoder(int batch_size, int num_species, int latent_dim, std::string filelocation){
    bsz      = batch_size;
    nspecies = num_species;
    nlatent  = latent_dim;

    // load models
    file_location = filelocation;
    std::string rhs_file = filelocation +"/neuralODE_rhs.ptc";
    std::string jac_file = filelocation + "/neuralODE_jac.ptc";
    std::string encoder_file = filelocation + "/neuralODE_encoder.ptc";
    std::string decoder_file = filelocation + "/neuralODE_decoder.ptc";


    model_rhs = torch::jit::load(rhs_file);
    model_rhs.eval();

    model_jac = torch::jit::load(jac_file);
    model_jac.eval();

    model_encoder = torch::jit::load(encoder_file);
    model_encoder.eval();

    model_decoder = torch::jit::load(decoder_file);
    model_decoder.eval();

    // allocate memory for the assign buffers
    abundance_buffer = (double *) malloc(sizeof(double)*bsz* nspecies);
    latent_buffer    = (double *) malloc(sizeof(double)*bsz* nspecies);
    init_abundance_buffer = (double *) malloc(sizeof(double)*bsz* nspecies);
    init_latent_buffer    = (double *) malloc(sizeof(double)*bsz* nlatent);
}

int EncoderDecoder::LoadAbundanceBuffer(std::string ic_file, unsigned long idx_start, int nsamples){
    readdata_from_hdf5(ic_file, nsamples, idx_start, &abundance_buffer[0]);
    abundance_transform(nsamples, &abundance_buffer[0]);
    std::memcpy(init_abundance_buffer, abundance_buffer, sizeof(double)*nspecies*nsamples);
    return 0;
}

int abundance_transform(int nsamples, double *abundance){
    // need to normalize to the density
    int idx_helium = 5;
    int nspecies = 10;
    double current_density;
    for (int i = 0; i < nsamples*nspecies; i++){
        //printf("density idx = %d", (i/nspecies + idx_helium));
        current_density = abundance[ (i/nspecies) * nspecies + idx_helium ]/0.24;
        //printf("abund[%d] = %0.5g; current_dens = %0.5g\n", i,  abundance[i], current_density);
        if ((i % nspecies != 5) && (i % nspecies != 9)){
            abundance[i] /= current_density;
        }
    }
    return 0;
}

int EncoderDecoder::EncodeToLatent(int nsamples, double *abundance_vector){
    // the goal of this function is to store the abundance vector, store its latent in the buffer
    // - populate init_abundance_buffer, init_latent_buffer

//    for (int k = 0; k < 10; k++)
//        printf("init_abund[%d] = %0.5g\n", k, abundance_buffer[k]);
    // 1. Copy the abundance to buffer
    std::memcpy(init_abundance_buffer, abundance_vector, (sizeof(double)* nsamples * nspecies));
    // 2. Apply in-place transform to the abundance-buffer
    abundance_transform(nsamples, &init_abundance_buffer[0]);

//    for (int k = 0; k < 10; k++)
//        printf("after init_abund[%d] = %0.5g\n", k, init_abundance_buffer[k]);

    torch::NoGradGuard no_grad;
    // 3. initialize a input_tensor
    std::vector<torch::jit::IValue> input_tensors;
    // 4. load initial abundance to inputs
    input_tensors.push_back(torch::from_blob(init_abundance_buffer, {nsamples, nspecies}, torch::kFloat64));
    
    //std::cout << input_tensors <<std::endl;
    // 5. forward pass to encode
    torch::Tensor output_tensor = model_encoder.forward(input_tensors).toTensor();

    // copy it to the latent_buffer
    //std::memcpy(latent_buffer, output_tensor.data_ptr(), sizeof(double)*nlatent*nsamples);
    std::memcpy(init_latent_buffer, output_tensor.data_ptr(), sizeof(double)*nlatent*nsamples);
    return 0;

}

int EncoderDecoder::LatentToReal(int nsamples, double *latent_ptr){
    
    torch::NoGradGuard no_grad;
    // initialize a input_tensor
    std::vector<torch::jit::IValue> inputs;
    // load initial abundance to inputs
    inputs.push_back(torch::from_blob(latent_ptr,         {nsamples, nlatent}, torch::kFloat64));
    inputs.push_back(torch::from_blob(init_latent_buffer,    {nsamples, nlatent}, torch::kFloat64));
    inputs.push_back(torch::from_blob(init_abundance_buffer, {nsamples, nspecies}, torch::kFloat64));
    // forward pass
    torch::Tensor output_tensor = model_decoder.forward(inputs).toTensor();
    // copy it to the latent_buffer
    std::memcpy(abundance_buffer, output_tensor.data_ptr(), sizeof(double)*nspecies*nsamples);
    return 0;

}
/*
int main(){

    int bsz        = 128;
    int nspecies   = 10;
    int latent_dim = 3;
    std::string model_loc = "/mnt/gv0/homes/kwoksun2/campus_cluster/test-enzo-dengo/cvode_enzo_64_1e-6";

    EncoderDecoder m1(bsz, nspecies, latent_dim, model_loc);

    void *ptr = static_cast<void *> (&m1);
    EncoderDecoder model_specification = * static_cast<EncoderDecoder *>(ptr);

    std::string filelocation = "/mnt/gv0/homes/kwoksun2/campus_cluster/test-enzo-dengo/cvode_enzo_64_1e-6";
    std::string ic_file = filelocation+"/new_dd0053_chemistry_5.hdf5"; 
    unsigned long idx_start  = 0;
    double *output_buffer = &model_specification.abundance_buffer[0];
    double *latent_buffer = &model_specification.latent_buffer[0];

    model_specification.LoadAbundanceBuffer(ic_file, idx_start, bsz);

    for (int k = 0; k < 1000; k++){
    
        idx_start += (unsigned long)bsz;
        //std::cout << k << " loop:" << std::endl;
    //unsigned long idx_start = 0;
    //std::string ic_file = model_loc +"/new_dd0053_chemistry_5.hdf5"; 
    model_specification.EncodeToLatent(bsz);
    model_specification.LatentToReal  (bsz);
    
    // now retrieve the class abundance buffer
   
    printf("OUPUT output: in main\n");
    for (int j = 0; j < 10; j++){
        std::cout << output_buffer[j] << " ";
        std::cout << std::endl;
    }

    
    // now retrieve the class abundance buffer
    printf("OUPUT latent: in main\n");
    for (int j = 0; j < 3; j++){
        std::cout << latent_buffer[j] << " ";
        std::cout << std::endl;
    }
    

    }
    return 0;
}
*/

/*
user_data initialize_user_data( std::string filelocation, int bsz, int nspecies ){
    user_data udata;
    udata.bsz = bsz;
    udata.nspecies = nspecies;

    std::string rhs_file = filelocation +"/neuralODE_rhs.ptc";
    std::string jac_file = filelocation + "/neuralODE_jac.ptc";
    std::string encoder_file = filelocation + "/neuralODE_encoder.ptc";
    std::string decoder_file = filelocation + "/neuralODE_decoder.ptc";


    udata.model_rhs = torch::jit::load(rhs_file);
    udata.model_rhs.eval();

    udata.model_jac = torch::jit::load(jac_file);
    udata.model_jac.eval();

    udata.model_encoder = torch::jit::load(encoder_file);
    udata.model_encoder.eval();

    udata.model_decoder = torch::jit::load(decoder_file);
    udata.model_decoder.eval();
    return udata;
}
*/

/////////////PYTORCH RHS //////////////////////////////////////
/*
static int f(realtype t, N_Vector y, N_Vector ydot, void *udata)
{
    // copy the N_Vector y to a preallocate input_tensors
    //
    user_data *my_data = (struct user_data *) udata;
    torch::jit::script::Module model;
    //model = my_data->model;

    double *y_ptr    =  NV_DATA_S(y);
    double *ydot_ptr =  NV_DATA_S(ydot);

    std::vector<torch::jit::IValue> input_tensors;
    input_tensors.push_back(torch::from_blob(y_ptr, {my_data->bsz, my_data->nspecies}, torch::kFloat64));

    // forward pass
    torch::Tensor output_tensor = model.forward(input_tensors).toTensor();
    //std::cout << output_tensor << std::endl;
    // copy the output
    std::memcpy(ydot_ptr, output_tensor.data_ptr(), sizeof(double)* my_data->nspecies* my_data->bsz);

    return(0);
}

static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, 
        void *udata, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{

    user_data *my_data = (struct user_data *) udata;

    // Get the Jacobian Model
    torch::jit::script::Module model_jac = my_data->model_jac;

    double *y_ptr    =  NV_DATA_S(y);
    std::vector<torch::jit::IValue> input_tensors;
    input_tensors.push_back(torch::from_blob(y_ptr, {my_data->bsz, my_data->nspecies}, torch::kFloat64));
    // forward pass to get the Jacobian
    torch::Tensor output_tensor = model_jac.forward(input_tensors).toTensor();

    //std::cout<< output_tensor  << std::endl;


    // access the data
    int bsz, nspecies, groupsize, group;
    bsz = my_data->bsz;
    nspecies = my_data->nspecies;
    groupsize = nspecies* nspecies;

    sunindextype *rowptrs = SUNSparseMatrix_IndexPointers(J);
    sunindextype *colvals = SUNSparseMatrix_IndexValues(J);
    realtype *data = SUNSparseMatrix_Data(J);
    SUNMatZero(J);

    //rowptrs[0] = 0;
    //rowptrs = &rowptrs[1];

    // initialize the index
    for (group = 0; group < bsz; group++){
        for (int r = 0; r < nspecies; r++){
            rowptrs[nspecies  * group + r] = groupsize *group+ r* nspecies;
            //printf("rowptr[%d] = %d\n", nspecies  * group + r, rowptrs[nspecies  * group + r]);
        }
        for (int e = 0; e < groupsize; e++){
            colvals[groupsize * group + e] = group * nspecies+ e % nspecies;
            //printf("colvals[%d] = %d\n", groupsize * group + e, colvals[groupsize * group + e]);

        }
    }
    rowptrs[nspecies  * bsz] = groupsize *bsz;

    // copy the results from the tensor
    std::memcpy( data, output_tensor.data_ptr(), sizeof(double)* groupsize * my_data->bsz);

    //SUNSparseMatrix_Print(J, stdout);

    return(0);
}
*/
