#include "latent_cvode_solver.h"


user_data initialize_user_data( std::string filelocation, int bsz, int nspecies ){
    user_data udata;
    udata.bsz = bsz;
    udata.nspecies = nspecies;

    torch::NoGradGuard no_grad;

    std::string rhs_file = filelocation +"/rhs_module.pt";
    std::string jac_file = filelocation + "/jac_module.pt";

    udata.model_rhs = torch::jit::load(rhs_file);
    udata.model_rhs.eval();

    udata.model_jac = torch::jit::load(jac_file);
    udata.model_jac.eval();


    std::vector<torch::jit::IValue> input_tensors;
    input_tensors.reserve(1);
    std::vector<torch::jit::IValue> tmult;
    tmult.reserve(1);

    return udata;
}


int LatentSolver::_RunSolver(double T1){
    double t = 0.0;
    int flag = CVode(cvode_mem, T1, y, &t, CV_NORMAL);
    //PrintFinalStats(this->cvode_mem);
    if (flag == CV_SUCCESS) return 0;
    if (check_flag(&flag, "CVode", 1)) return -1;
}

LatentSolver::~LatentSolver(){
    N_VDestroy(y);
    N_VDestroy(abstol);
    /* Free integrator memory */
    CVodeFree(&cvode_mem);
    /* Free the linear solver memory */
    SUNLinSolFree(LS);
    /* Free the matrix memory */
    SUNMatDestroy(A);
}


// class initializer for the solver object
LatentSolver::LatentSolver(int batch_size, int num_species, double relative_tolerance, double *data_pointer, double *tmult_pointer, void* udata)
    : bsz(batch_size), nspecies(num_species), reltol(relative_tolerance), data_ptr(data_pointer)
{
    int flag;

    
    //bsz      = batch_size;
    //nspecies = num_species;
    NEQ      = bsz * nspecies;
    //reltol   = relative_tolerance;
    //data_ptr = data_pointer;

    
    //std::string filelocation = "/mnt/gv0/homes/kwoksun2/campus_cluster/test-enzo-dengo/cvode_enzo_64_1e-6";
    //user_data mydata = initialize_user_data(filelocation, bsz, nspecies);
    user_data *mydata =  (user_data *)(udata);

    y  = N_VNew_Serial(NEQ);
    check_flag((void *)y, "N_VNew_Serial", 0);
    abstol = N_VNew_Serial(NEQ); 
    check_flag((void *)abstol, "N_VNew_Serial", 0);
    
    // expose the y pointer to the class structure
    latent_ptr = NV_DATA_S(y);
    
    // read the data into the N_Vector space
    this->LoadN_Vector(bsz, data_ptr);

    // initialize tmult 

    // point the tensor to the memory space of the cvode internal solver
    mydata->input_tensors.push_back(torch::from_blob(latent_ptr, {batch_size, num_species}, torch::kFloat64));
    mydata->input_tensors.push_back(torch::from_blob(tmult_pointer, {batch_size, 1}, torch::kFloat64));
    cvode_mem = CVodeCreate(CV_BDF);


    //if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

    /* Call CVodeInit to initialize the integrator memory and specify the
     * user's right hand side function in y'=f(t,y), the inital time T0, and
     * the initial dependent variable vector y. */
    flag = CVodeInit(cvode_mem, f, T0, y);
    //if (check_flag(&flag, "CVodeInit", 1)) return(1);

    /* Call CVodeSetUserData to attach the user data structure */
    flag = CVodeSetUserData(cvode_mem, udata);
    //if (check_flag(&flag, "CVodeSetUserData", 1)) return(1);
    /* Call CVodeSVtolerances to specify the scalar relative tolerance
     * and vector absolute tolerances */
    flag = CVodeSVtolerances(cvode_mem, reltol, abstol);
    check_flag(&flag, "CVodeSVtolerances", 1);

    /* Create dense SUNMatrix for use in linear solves */
    //A = SUNDenseMatrix(NEQ, NEQ);
    //if(check_flag((void *)A, "SUNDenseMatrix", 0)) return(1);
    /* Create Sparse SUNMatrix for use in linear solve*/
    int nnz = bsz * nspecies * nspecies;
    A = SUNSparseMatrix(bsz* nspecies, bsz*nspecies, nnz, CSR_MAT);
    //if(check_flag((void *)A, "SUNSparseMatrix", 0)) return(1);
    /* Create dense SUNLinearSolver object for use by CVode */
    //LS = SUNDenseLinearSolver(y, A);
    //if(check_flag((void *)LS, "SUNDenseLinearSolver", 0)) return(1);
    /* Create KLU solver object for use by CVode */
    LS = SUNLinSol_KLU(y, A);
    //if(check_flag((void *)LS, "SUNLinSol_KLU", 0)) return(1);

    /* Call CVDlsSetLinearSolver to attach the matrix and linear solver to CVode */
    flag = CVodeSetLinearSolver(cvode_mem, LS, A);
    //if(check_flag(&flag, "CVodeSetLinearSolver", 1)) return(1);

    /* Set the user-supplied Jacobian routine Jac */
    flag = CVodeSetJacFn(cvode_mem, Jac);
    //if(check_flag(&flag, "CVodeSetJacFn", 1)) return(1);
}

int LatentSolver::LoadN_Vector(int nsamples, double *latent_z_ptr){
    // populate the y, and abstol;
    // the initial condition z0 from the encoder 
    realtype *y_ptr    =  NV_DATA_S(y);
    realtype *abs_ptr  =  NV_DATA_S(abstol);
    std::memcpy(y_ptr, latent_z_ptr, sizeof(double)* nsamples * nspecies);
    for (int i = 0; i < nsamples*nspecies; i++){
        abs_ptr[i] = reltol * fabs(y_ptr[i]);
    }
    //printf("abs+ptr[0] = %0.5g\n", abs_ptr[0]);
    
    return 0;
}

int LatentSolver::RunSolver(int nsamples, double *latent_z_ptr, double T1){
    this->LoadN_Vector(nsamples, latent_z_ptr);
    int flag = CVodeReInit(cvode_mem, 0.0, y);
    this->_RunSolver(T1);
}


/////////////PYTORCH RHS //////////////////////////////////////
static int f(realtype t, N_Vector y, N_Vector ydot, void *udata)
{
    // copy the N_Vector y to a preallocate input_tensors
    //
    //
    torch::NoGradGuard no_grad;
    user_data *my_data = (struct user_data *)(udata);
    torch::jit::script::Module model = my_data->model_rhs;

    double *ydot_ptr =  NV_DATA_S(ydot);
    
    //std::vector<torch::jit::IValue> input_tensors = my_data->input_tensors;
    //input_tensors.shrink_to_fit();
    //input_tensors.emplace_back(torch::from_blob(y_ptr, {my_data->bsz, my_data->nspecies}, torch::kFloat64));

    torch::Tensor output_tensor = model.forward( my_data->input_tensors).toTensor();
    //std::cout << output_tensor << std::endl;
    // copy the output
    std::memcpy(ydot_ptr, output_tensor.data_ptr(), sizeof(double)* my_data->nspecies* my_data->bsz);
    //N_VPrint_Serial(y);

    return(0);
}

static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, 
        void *udata, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{

    //std::string filelocation = "/mnt/gv0/homes/kwoksun2/campus_cluster/test-enzo-dengo/cvode_enzo_64_1e-6";
    //std::string rhs_file = filelocation +"/neuralODE_jac.ptc";
    //torch::jit::script::Module model_jac = torch::jit::load(rhs_file);
    
    user_data *my_data = (struct user_data *)(udata);

    torch::NoGradGuard no_grad;
    // Get the Jacobian Model
    torch::jit::script::Module model_jac = my_data->model_jac;

    double *y_ptr    =  NV_DATA_S(y);
    
    //std::vector<torch::jit::IValue> input_tensors = my_data->input_tensors;
    //input_tensors.shrink_to_fit();
    //input_tensors.emplace_back(torch::from_blob(y_ptr, {my_data->bsz, my_data->nspecies}, torch::kFloat64));
    // forward pass to get the Jacobian
    torch::Tensor output_tensor = model_jac.forward(my_data->input_tensors).toTensor();


    // access the data
    int bsz, nspecies, groupsize, group;
    bsz = my_data->bsz;
    nspecies = my_data->nspecies;
    groupsize = nspecies* nspecies;

    sunindextype *rowptrs = SUNSparseMatrix_IndexPointers(J);
    sunindextype *colvals = SUNSparseMatrix_IndexValues(J);
    realtype *data = SUNSparseMatrix_Data(J);

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


/*
int main(){
    // initialize our latent solver
    int batch_size = 2;
    int num_species = 3;
    double reltol   = 1e-5;

    // initialize a random data pointer
    double latent_z[batch_size * num_species];
    for (int i = 0; i < batch_size*num_species; i++){
        if (i%3 == 0)
            latent_z[i] = -2.3541; 
        if (i%3 == 1)
            latent_z[i] = 1.3025;
        if (i%3 == 2)
            latent_z[i] = -0.5458;
    }
    
    std::string filelocation = "/mnt/gv0/homes/kwoksun2/campus_cluster/test-enzo-dengo/cvode_enzo_64_1e-6";
    user_data mydata = initialize_user_data(filelocation, batch_size, num_species);

    LatentSolver solver(batch_size, num_species, reltol, latent_z, (void*) &mydata);

    //solver.LoadN_Vector(batch_size, latent_z);

    double tout = 1.3742;
    solver.RunSolver(tout);

    // reinit and then rerun


    for (int k = 0; k < 10; k++)
    solver.RerunSolver(batch_size, latent_z, 0 );

    // now get the pointers

    realtype *y_ptr    =  NV_DATA_S(solver.y);
    for (int i = 0; i < 3; i++){
        printf("y[%d] = %0.5g\n", i, y_ptr[i]);
    }
    PrintFinalStats(solver.cvode_mem);

}
*/

static int check_flag(void *flagvalue, const char *funcname, int opt)
{
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == NULL) {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return(1); }

    /* Check if flag < 0 */
    else if (opt == 1) {
        errflag = (int *) flagvalue;
        if (*errflag < 0) {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
            return(1); }}

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && flagvalue == NULL) {
        fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return(1); }

    return(0);
}

static void PrintFinalStats(void *cvode_mem)
{
    long int nst, nfe, nsetups, nje, nfeLS, nni, ncfn, netf, nge;
    int flag;

    flag = CVodeGetNumSteps(cvode_mem, &nst);
    check_flag(&flag, "CVodeGetNumSteps", 1);
    flag = CVodeGetNumRhsEvals(cvode_mem, &nfe);
    check_flag(&flag, "CVodeGetNumRhsEvals", 1);
    flag = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
    check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
    flag = CVodeGetNumErrTestFails(cvode_mem, &netf);
    check_flag(&flag, "CVodeGetNumErrTestFails", 1);
    flag = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
    check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
    flag = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
    check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);

    flag = CVDlsGetNumJacEvals(cvode_mem, &nje);
    check_flag(&flag, "CVDlsGetNumJacEvals", 1);
    flag = CVDlsGetNumRhsEvals(cvode_mem, &nfeLS);
    check_flag(&flag, "CVDlsGetNumRhsEvals", 1);

    flag = CVodeGetNumGEvals(cvode_mem, &nge);
    check_flag(&flag, "CVodeGetNumGEvals", 1);

    printf("\nFinal Statistics:\n");
    printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld nfeLS = %-6ld nje = %ld\n",
            nst, nfe, nsetups, nfeLS, nje);
    printf("nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n \n",
            nni, ncfn, netf, nge);
}

