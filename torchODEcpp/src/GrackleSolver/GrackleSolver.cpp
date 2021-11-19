/*
 *Taken Primarily from Grackle Example
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include "sys/time.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <string>

#include "hdf5.h"
#include "hdf5_hl.h"

#define mh     1.67262171e-24
#define kboltz 1.3806504e-16

extern "C" {
#include <grackle.h>
}
int main(int argc, char* argv[])
{

  /*********************************************************************
  / Initial setup of units and chemistry objects.
  / This should be done at simulation start.
  *********************************************************************/

  // Set initial redshift (for internal units).
  double initial_redshift = 0.;

  // First, set up the units system.
  // These are conversions from code units to cgs.
  code_units my_units;
  my_units.comoving_coordinates = 0;    // 1 if cosmological sim, 0 if not
  my_units.density_units        = 1.0;
  my_units.length_units         = 1.0;
  my_units.time_units           = 1.0;
  my_units.velocity_units       = my_units.length_units / my_units.time_units;
  my_units.a_units              = 1.0;  // units for the expansion factor
  // Set expansion factor to 1 for non-cosmological simulation.
  my_units.a_value = 1. / (1. + initial_redshift) / my_units.a_units;

  // Second, create a chemistry object for parameters.  This needs to be a pointer.
  chemistry_data *my_grackle_data;
  my_grackle_data = new chemistry_data;
  if (set_default_chemistry_parameters(my_grackle_data) == 0) {
    fprintf( stderr, "Error in set_default_chemistry_parameters.\n" );
    return 0;
  }

  // Set parameter values for chemistry.
  // Access the parameter storage with the struct you've created
  // or with the grackle_data pointer declared in grackle.h.
  grackle_data->use_grackle            = 1;          // chemistry on
  grackle_data->with_radiative_cooling = 1;          // cooling on
  grackle_data->primordial_chemistry   = 2;          // fully tabulated cooling
  grackle_data->metal_cooling          = 0;          // metal cooling on
  grackle_data->dust_chemistry         = 0;          // dust processes
  grackle_data->UVbackground           = 0;          // UV background on
  grackle_data->grackle_data_file      = "CloudyData_UVB=HM2012.h5"; // data file
# ifdef _OPENMP
  int NThread=omp_get_max_threads();
  grackle_data->omp_nthreads           = NThread; // NThread;    // number of OpenMP threads
  printf("with omp threads %d\n", NThread);
                                                    // (remove this line if Grackle is not compiled with OpenMP support)
# endif

  // Finally, initialize the chemistry object.
  if ( initialize_chemistry_data(&my_units) == 0 ) {
    fprintf( stderr, "Error in initialize_chemistry_data.\n" );
    return EXIT_FAILURE;
  }


    hid_t file_id;
    hsize_t dims[1];
    

    if (argc != 3){
        printf("\n no input file is given, frac_tff \n");
        return 1;
    }

    double frac_tff = atof(argv[2]);
    

    file_id = H5Fopen(argv[1], H5F_ACC_RDWR, H5P_DEFAULT);
    H5LTget_dataset_info(file_id, "/ge", dims, NULL, NULL);
    printf("ncells = %lu\n", dims[0]);
    printf("file in process: %s\n", argv[1]);

    int nsteps = 75;
    
    double *H2I_Density  = (double *) malloc(sizeof(double)* dims[0] * nsteps);
    double *H2II_Density = (double *) malloc(sizeof(double)* dims[0] * nsteps);
    double *HI_Density  = (double *) malloc(sizeof(double)* dims[0]  * nsteps);
    double *HII_Density = (double *) malloc(sizeof(double)* dims[0]  * nsteps);
    double *HM_Density  = (double *) malloc(sizeof(double)* dims[0]  * nsteps);
    double *HeI_Density   = (double *) malloc(sizeof(double)* dims[0] *nsteps);
    double *HeII_Density  = (double *) malloc(sizeof(double)* dims[0] *nsteps);
    double *HeIII_Density = (double *) malloc(sizeof(double)* dims[0] *nsteps);
    double *de_Density = (double *) malloc(sizeof(double)* dims[0] * nsteps);
    double *ge_Density = (double *) malloc(sizeof(double)* dims[0] * nsteps);
    double *density = (double *) malloc(sizeof(double)* dims[0] *nsteps);


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


    grackle_verbose=1;

    // initialize the dengo ffield data
    grackle_field_data *field_data = (grackle_field_data *) malloc(sizeof(grackle_field_data));
    
    double mH = 1.67e-24;
    double k  = 1.380e-16;
    double tiny = 1.0e-40; 
    
    double G = 6.67259e-8;
    code_units *units = (code_units *) malloc(sizeof(code_units));
    units->density_units = 1.0;
    units->length_units = 1.0;
    units->time_units = 1.0;
    units->velocity_units = 1.0; 
    
    
    field_data->e_density   = (double *) &de_Density[0];
    field_data->HI_density  = (double *) &HI_Density[0];
    field_data->HeI_density = (double *) &HeI_Density[0];
    field_data->HII_density = (double *) &HII_Density[0];
    field_data->HeIII_density = (double *) &HeIII_Density[0];
    field_data->H2I_density   = (double *) &H2I_Density[0];
    field_data->H2II_density  = (double *) &H2II_Density[0];
    field_data->internal_energy = (double *) &ge_Density[0];
    field_data->HeII_density    = (double *) &HeII_Density[0];
    field_data->HM_density      = (double *) &HM_Density[0];
    field_data->density         = (double *) &density[0];

    // 
    int gstart[3];
    int gend[3];
    int gd[3];

    gstart[0] = 0;
    gstart[1] = 0;
    gstart[2] = 0;

    gend[0] = 2-1;
    gend[1] = 4-1;
    gend[2] = 312757-1;

    gd[0] = 2;
    gd[1] = 4;
    gd[2] = 312757;

    field_data->grid_start = &gstart[0];
    field_data->grid_end   = &gend[0];
    field_data->grid_dimension = &gd[0];
    
    //field_data->floor_value = 1e-40;
    double dtf = 1.3742e11; 
    double dt = dtf/1e4; // frac_tff * pow(density[0]*G, -0.5)/ my_units.time_units;

  unsigned long field_size =dims[0];

  fprintf(stderr, "H2I[0] = %0.5g\n", H2I_Density[0]/ density[0]);
    
  // measure time too
    
    double start, end;
    double cpu_time_used;
    start = omp_get_wtime();

   for (int k = 1; k < nsteps; k++){
       // integrate the system
       if (solve_chemistry(&my_units, field_data, dt) == 0) {
           fprintf(stderr, "Error in solve_chemistry.\n");
           return EXIT_FAILURE;
        }
       dt *= 1.1;
        // copy the data to the next memspace
        memcpy( &H2I_Density[k*dims[0]], &H2I_Density[0], sizeof(double)*dims[0]); 
        memcpy( &H2II_Density[k*dims[0]], &H2II_Density[0], sizeof(double)*dims[0]); 
        memcpy( &HI_Density[k*dims[0]], &HI_Density[0], sizeof(double)*dims[0]); 
        memcpy( &HII_Density[k*dims[0]], &HII_Density[0], sizeof(double)*dims[0]); 
        memcpy( &HM_Density[k*dims[0]], &HM_Density[0], sizeof(double)*dims[0]); 
        memcpy( &HeI_Density[k*dims[0]], &HeI_Density[0], sizeof(double)*dims[0]); 
        memcpy( &HeII_Density[k*dims[0]], &HeII_Density[0], sizeof(double)*dims[0]); 
        memcpy( &HeIII_Density[k*dims[0]], &HeIII_Density[0], sizeof(double)*dims[0]); 
        memcpy( &de_Density[k*dims[0]], &de_Density[0], sizeof(double)*dims[0]); 
        memcpy( &ge_Density[k*dims[0]], &ge_Density[0], sizeof(double)*dims[0]); 
        memcpy( &density[k*dims[0]], &density[0], sizeof(double)*dims[0]); 
   }

    end = omp_get_wtime();
    cpu_time_used = ((double) (end - start)) ;
    printf("fun() took %f seconds to execute \n", cpu_time_used);

  fprintf(stderr, "density = %0.5g; dtf = %0.5g\n", density[0], dt);

  fprintf(stderr, "H2I[0] = %0.5g\n", H2I_Density[0]/ density[0]);

    // write all the output to file
    std::string filename = "grackle_sol_" + std::to_string(frac_tff) + ".hdf5";
    std::cout << filename <<std::endl;
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    hsize_t dims2d[1] = {dims[0]*nsteps};
    H5LTmake_dataset_double(file_id, "/H2I", 1, dims2d, H2I_Density );
    H5LTmake_dataset_double(file_id, "/H2II", 1, dims2d, H2II_Density );
    H5LTmake_dataset_double(file_id, "/HI", 1, dims2d, HI_Density );
    H5LTmake_dataset_double(file_id, "/HII", 1, dims2d, HII_Density );
    H5LTmake_dataset_double(file_id, "/HM", 1, dims2d, HM_Density );
    H5LTmake_dataset_double(file_id, "/HeI", 1, dims2d, HeI_Density );
    H5LTmake_dataset_double(file_id, "/HeII", 1, dims2d, HeII_Density );
    H5LTmake_dataset_double(file_id, "/HeIII", 1, dims2d, HeIII_Density );
    H5LTmake_dataset_double(file_id, "/de", 1, dims2d, de_Density );
    H5LTmake_dataset_double(file_id, "/ge", 1, dims2d, ge_Density );

    H5LTset_attribute_double(file_id, ".", "time_taken", &cpu_time_used, 1);

    H5Fclose(file_id);


  return 0;
}
