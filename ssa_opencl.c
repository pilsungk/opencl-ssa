/**
 *  FILE:    ssa_opencl.c
 *
 *  AUTHOR:  Pilsung Kang
 *  CREATED: January 6, 2019
 *  LAST MODIFIED: Jan. 11, 2019
 *             BY: Pilsung Kang
 *             TO: complete random number generation 
 *
 *  SUMMARY: Stochastic simulation algorithm in OpenCL
 *
 *  NOTES: 
 *      Adapted from the hello world example code by Apple
 *      Used TinyMT RNG v1.1.1 
 *      - http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/TINYMT/index.html
 *
 *  TO DO: port CUDA ssa version into OpenCL
 *
 *  Version:    <1.0>
 */
 

////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
//#include <math.h>
#include <time.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS  // suppress deprecation warning for clCreateCommandQueue
//#include <CL/opencl.h>

#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

////////////////////////////////////////////////////////////////////////////////

/* problem parameters and cuda threads launch geometry */
#include "prob_params.h"

#define PROGRAM_FILE "ssa_kernel.cl"
#define KERNEL_FUNC "ssa_kernel"

static const int x[NX] = {1200, 600, 0};

static void init_x_array(int *xarr)
{
    for (int i=0; i<NTHREADS; i++)
        for (int j=0; j<NX; j++) 
            xarr[NX*i+j] = x[j];
}

// CL_CHECK copied from http://svn.clifford.at/tools/trunk/examples/cldemo.c
#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     abort();                                                                   \
   } while (0)



/* Create program from a file and compile it */
// copied from OpenCL in Action 
// https://github.com/jeremyong/opencl_in_action/blob/master/Ch11/bsort8/bsort8.c
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err;

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if(program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file */
    program = clCreateProgramWithSource(ctx, 1, 
            (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
                0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
                log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
    size_t localsize, globalsize;
      
    unsigned int numWorkItems = NTHREADS;      // total number of work-items

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue queue;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem x_array_d;                       // device memory used for the input array
    cl_mem finalT_array_d;                      // device memory used for the output array
    
	cl_platform_id platforms[100];
	cl_uint platforms_n = 0;
	CL_CHECK(clGetPlatformIDs(100, platforms, &platforms_n));

#if 0
	printf("=== %d OpenCL platform(s) found: ===\n", platforms_n);
	for (int i=0; i<platforms_n; i++)
	{
		char buffer[10240];
		printf("  -- %d --\n", i);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 10240, buffer, NULL));
		printf("  PROFILE = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 10240, buffer, NULL));
		printf("  VERSION = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, buffer, NULL));
		printf("  NAME = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL));
		printf("  VENDOR = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL));
		printf("  EXTENSIONS = %s\n", buffer);
	}
#endif

	if (platforms_n == 0)
		return 1;

	cl_device_id devices[100];
	cl_uint devices_n = 0;
	CL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 100, devices, &devices_n));

#if 0
	printf("=== %d OpenCL device(s) found on platform:\n", devices_n);
	for (int i=0; i<devices_n; i++)
	{
		char buffer[10240];
		cl_uint buf_uint;
		cl_ulong buf_ulong;
		printf("  -- %d --\n", i);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
		printf("  DEVICE_NAME = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
		printf("  DEVICE_VENDOR = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
		printf("  DEVICE_VERSION = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));
		printf("  DRIVER_VERSION = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL));
		printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL));
		printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));
		printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
	}
#endif

	if (devices_n == 0)
		return 1;

    // Connect to a compute device
    //
    device_id = devices[0];
    // pilsung - this call is redundant - clGetDeviceIDs was called above
    CL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 1, &device_id, NULL));

    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context! Error code %d\n", err);
        return EXIT_FAILURE;
    }

    // Create a command queue
    //
    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!queue)
    {
        printf("Error: Failed to create a command queue!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source 
    //
    program = build_program(context, device_id, PROGRAM_FILE);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    x_array_d = clCreateBuffer(context,  CL_MEM_READ_WRITE, sizeof(int)*NX*NTHREADS, NULL, NULL);
    if (!x_array_d)
    {
        printf("Error: Failed to allocate device memory (x_array_d)!\n");
        exit(1);
    }    

    finalT_array_d = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float)*NTHREADS, NULL, NULL);
    if (!finalT_array_d)
    {
        printf("Error: Failed to allocate device memory (finalT_array_d)!\n");
        exit(1);
    }    

    int* x_array_h = (int*) malloc(NTHREADS*NX*sizeof(int));
    init_x_array(x_array_h);
    clEnqueueWriteBuffer(queue, x_array_d, CL_TRUE, 0, NTHREADS*NX*sizeof(int), x_array_h, 0, NULL, NULL); 

    float* finalT_array_h = (float*) malloc(NTHREADS*sizeof(float));

    int* counter_array_h = (int*) malloc(NTHREADS*sizeof(int));
    cl_mem counter_array_d;                       // device memory used for the input array
    counter_array_d = clCreateBuffer(context,  CL_MEM_READ_WRITE, sizeof(int)*NTHREADS, NULL, NULL);

    // Set the arguments to our compute kernel
    //
    unsigned int seed = (unsigned) time(NULL);
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &x_array_d);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem),(void*) &finalT_array_d);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int),(void*) &numWorkItems);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int),(void*) &seed);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem),(void*) &counter_array_d);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    // err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, 
    //                                  sizeof(localsize), &localsize, NULL);
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to retrieve kernel work group info! %d\n", err);
    //     exit(1);
    // }

    // Number of total work items - localSize must be devisor
    //globalsize = ceil(numWorkItems/(float)localsize)*localsize;

    localsize = XBLOCKSIZE;
    globalsize = numWorkItems;
    printf("global size=%lu, local size=%lu\n", globalsize, localsize);



    /* Enqueue kernel with profiling event   */
    cl_event kernel_completion;
    cl_ulong time_start, time_end;

    //printf("Random numbers generated inside kernel:\n");
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalsize, &localsize, 0, NULL, &kernel_completion); 
    clFinish(queue);

    CL_CHECK(clWaitForEvents(1, &kernel_completion));

    CL_CHECK(clGetEventProfilingInfo(kernel_completion, CL_PROFILING_COMMAND_START,
               sizeof(time_start), &time_start, NULL));
    CL_CHECK(clGetEventProfilingInfo(kernel_completion, CL_PROFILING_COMMAND_END,
               sizeof(time_end), &time_end, NULL));

    double exe_time = time_end - time_start;
    printf("Kernel exec time = %.3f msec\n", exe_time/1000000.0);

    /* Read the kernel's output    */
    clEnqueueReadBuffer(queue, x_array_d, CL_TRUE, 0, NX*NTHREADS*sizeof(int), x_array_h, 0, NULL, NULL); 
    clEnqueueReadBuffer(queue, finalT_array_d, CL_TRUE, 0, NTHREADS*sizeof(float), finalT_array_h, 0, NULL, NULL); 
    clEnqueueReadBuffer(queue, counter_array_d, CL_TRUE, 0, NTHREADS*sizeof(int), counter_array_h, 0, NULL, NULL); 

#if 0
    //printf("numbers returned to host:\n");
    for (int i=0; i<numWorkItems; i++) {
        for (int j=0; j<NX; j++) {
            printf("%d ", x_array_h[i*NX+j]);
        }
        printf(", %d, ", counter_array_h[i]);
        printf("%f\n", finalT_array_h[i]);
    }
#endif

    // Shutdown and cleanup
    //
    free(x_array_h);
    free(finalT_array_h);
    CL_CHECK(clReleaseEvent(kernel_completion));
    CL_CHECK(clReleaseMemObject(x_array_d));
    CL_CHECK(clReleaseMemObject(finalT_array_d));
    CL_CHECK(clReleaseProgram(program));
    CL_CHECK(clReleaseKernel(kernel));
    CL_CHECK(clReleaseCommandQueue(queue));
    CL_CHECK(clReleaseContext(context));

    return 0;
}

