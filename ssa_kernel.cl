#include "prob_params.h"
#include "TinyMT/opencl/tinymt32_jump.clh"


/// example OpenCL kernel
//
__kernel void example( __global int* output, const unsigned int count, unsigned int seed) 
{  
    size_t i = get_global_id(0);   
    if(i < count) { 
#if 0
        tinymt32j_t tinymt;
        tinymt32j_init_jump(&tinymt, (i+seed)); //init rng to do something somewhat random within each thread
        //unsigned int val = tinymt32j_uint32(&tinymt);
        float val = tinymt32j_single01(&tinymt);

        output[i] = val;
#endif 

        output[i] = get_local_id(0);
    }                                                
}                                                  

/// ssa kernel CUDA version
//
__kernel void ssa_kernel(__global int* x, __global float* ftime, 
                         const unsigned int count, const unsigned int seed, __global int* counters)
{
    size_t tid = get_global_id(0);   

    __local int xShared[NX*XBLOCKSIZE];  // shared mem is per-blcok
    __local int nu[DIMX_NU][DIMY_NU];
    __local float proprates[NCHANNEL];
    __local int done[XBLOCKSIZE]; // XBLOCKSIZE == blockDim.x == local_size == get_local_size(0)

    const int xBegin = NX * tid;
    int tx = get_local_id(0);
    const int xSharedBegin = tx * NX;

    if (tx == 0) {
        nu[0][0] = -1; nu[0][1] = 1;  nu[0][2] = 0;
        nu[1][0] = 1;  nu[1][1] = -1; nu[1][2] = -1;
        nu[2][0] = 0;  nu[2][1] = 0;  nu[2][2] = 1;

        proprates[0] = 1.0f;
        proprates[1] = 2.0f;
        proprates[2] = 0.00005f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i=0; i<NX; i++) xShared[xSharedBegin+i] = x[xBegin+i];
    
    float curTime = 0.0f;
    
    float a0, a[3];
    float f, jsum, tau;
    int rxn;
    float rand1, rand2;
    int total_done;
    int counter = 0;

    done[tx] = 0;

    tinymt32j_t tinymt;
    tinymt32j_init_jump(&tinymt, (tid+seed)); //init rng to do something somewhat random within each thread
    rand1 = tinymt32j_single01(&tinymt);
    rand2 = tinymt32j_single01(&tinymt);

    while (rand1 < ALMOST_ZERO || rand2 < ALMOST_ZERO) {
        rand1 = tinymt32j_single01(&tinymt);
        rand2 = tinymt32j_single01(&tinymt);
    }

    while (1) {
        counter++;

        if (!done[tx]) {
            // take step -- 1. choose the channel to fire
            //printf("x0 = %d, x1 = %d, x2 = %d\n", x[xBegin], x[xBegin+1], x[xBegin+2]);
            a[0] = xShared[xSharedBegin]*proprates[0];
            a[1] = xShared[xSharedBegin+1]*proprates[1];
            a[2] = xShared[xSharedBegin+1]*proprates[2];

            a0 = a[0] + a[1] + a[2];
            f = rand1 * a0;

            jsum = 0.0;

            for(rxn=0; jsum < f; rxn++) jsum += a[rxn];
            rxn--;


            // take step -- 2. fire the chosen channel
            for (int i=0; i<NX; i++) {
                xShared[xSharedBegin+i] += nu[i][rxn];
            }

            // take step -- 3. calculate the time step
            tau = -log(rand2) / a0;
            curTime += tau;

            // negative state check
            for (int i=0; i<NX; i++) {
                if (xShared[xSharedBegin+i] < 0) {
                    for (int j=0; j<NX; j++) { 
                        xShared[xSharedBegin+j] -= nu[j][rxn];
                    }

                    curTime -= tau;
                    break;
                }
            }

            if (curTime > FINALTIME) done[tx] = 1;

            rand1 = tinymt32j_single01(&tinymt);
            rand2 = tinymt32j_single01(&tinymt);
            while (rand1 < ALMOST_ZERO || rand2 < ALMOST_ZERO) {
                rand1 = tinymt32j_single01(&tinymt);
                rand2 = tinymt32j_single01(&tinymt);
            }
        }
        if (done[tx]) break;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    ftime[tid] = curTime;
    counters[tid] = counter;
    for (int i=0; i<NX; i++) x[xBegin+i] = xShared[xSharedBegin+i];
}

