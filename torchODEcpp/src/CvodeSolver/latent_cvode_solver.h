#include <stdio.h>

#include <cvode/cvode.h>               /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <cvode/cvode_direct.h>        /* access to CVDls interface            */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype      */
#include <sunmatrix/sunmatrix_sparse.h> /* access to sparse SUNMatrix           */
#include <sunlinsol/sunlinsol_klu.h>    /* access to KLU sparse direct solver   */

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <random>
#include <ctime>

// data structure to be passed around in the solver

struct user_data{
    int bsz;
    int nspecies;
    torch::jit::script::Module model_rhs;
    torch::jit::script::Module model_jac;
    torch::jit::script::Module model_encoder;
    torch::jit::script::Module model_decoder;

    std::vector<torch::jit::IValue> input_tensors;
};


class LatentSolver{

    private:
        int NEQ;
        const int bsz;
        const int nspecies;
        const double reltol;
        const double T0 = 0.0;
        N_Vector y;
        N_Vector abstol;
        N_Vector ydot;
        SUNMatrix A;
        SUNLinearSolver LS;
        void *cvode_mem;
        double *data_ptr;
        double *latent_ptr;

    public:
        LatentSolver(int batch_size, int num_species, double reltol, double *data_pointer, void* udata);
        ~LatentSolver();
        int _RunSolver(double tout);
        int LoadN_Vector(int nsamples, double *latent_z_ptr);
        int RunSolver(int nsamples, double *latent_z_ptr, double T1);
        double *get_latent_ptr(){return this->latent_ptr;}

};

static int check_flag(void *flagvalue, const char *funcname, int opt);
static void PrintFinalStats(void *cvode_mem);
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *udata, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int f(realtype t, N_Vector y, N_Vector ydot, void *udata);

user_data initialize_user_data( std::string filelocation, int bsz, int nspecies );
