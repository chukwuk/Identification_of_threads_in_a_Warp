#include <cuda_runtime.h>

#ifndef __THREADSINWARP_H
#define __THREADSINWARP_H


struct threadProperties {
    unsigned long long time;
    int thread_x;
    int thread_y;
    int thread_z;
    int value;
};
	



__global__  void threadsInWarp3D(threadProperties* threadsDev, int* globalData); 


#endif
