#ifndef TORCH_RHS_JAC_H_
#define TORCH_RHS_JAC_H_

#include "../ReadIO/read_hdf5.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>


class EncoderDecoder{
    private:
        int bsz;
        int nspecies;
        int nlatent;
        int idx_helium = 5;
        double T1;
        double frac_helium = 0.24;
        double G = 6.67e-8;
        std::string file_location;
        torch::jit::script::Module model_rhs;
        torch::jit::script::Module model_jac;
        torch::jit::script::Module model_encoder;
        torch::jit::script::Module model_decoder;
        
        double *latent_buffer;
        double *abundance_buffer;
        double *init_latent_buffer;
        double *init_abundance_buffer;
        double *invtff_buffer;

    public:
        // Constructor
        EncoderDecoder(int batch_size, int nspecies, int latent_dim, double T1, std::string filelocation);
        //~EncoderDecoder(){delete[] abundance_buffer; delete[] latent_buffer;};

        int LoadAbundanceBuffer(std::string ic_file, unsigned long idx_start, int nsamples);
        int EvaluateInvTff(int nsamples, double T1,double *abund_ptr);

        int EncodeToLatent(int nsamples, double *abund_ptr);
        int LatentToReal  (int nsamples, double *latent_ptr);
        double *get_latent_ptr(){return this->latent_buffer;};
        double *get_abund_ptr(){return this->abundance_buffer;};
        double *get_init_latent_ptr(){return this->init_latent_buffer;};
        double *get_init_abund_ptr(){return this->init_abundance_buffer;};
        double *get_invtff_ptr(){return this->invtff_buffer;};
};

int abundance_transform(int nsamples, double *abundance);
#endif // TORCH_RHS_JAC_H_
