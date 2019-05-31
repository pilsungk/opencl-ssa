#ifndef PROB_PARAMS_H
#define PROB_PARAMS_H


// 1. 256 blocks * 32 threads = 8192
// 2. 4096 (256*16) blocks * 32 threads = 131072
// 3. 32768 (256*16*8) blocks * 32 threads = 1048576
// 4. 32768 (256*16*8) blocks * (32*8) threads = 8388608 ==> simul is not stopping (for 7 days)
#define XBLOCKSIZE (32)       // cuda thread block x
#define YBLOCKSIZE 1       // cuda thread block y
#define XGRIDSIZE  (32768)       // cuda grid size x ==> # groups
#define YGRIDSIZE  1        // cuda grid size y
#define NTHREADS   ((XBLOCKSIZE)*(YBLOCKSIZE)*(XGRIDSIZE)*(YGRIDSIZE))

// Input Problem Constants
#define NX 3                // number of spicies
#define FINALTIME 1000.0    // time when the evolution finishes 

#define NCHANNEL 3
#define DIMX_NU NCHANNEL
#define DIMY_NU NX

#define ALMOST_ZERO (1e-19)

#endif 
