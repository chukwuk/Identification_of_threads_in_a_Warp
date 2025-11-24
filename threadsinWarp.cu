#include <math.h>
#include "threadsinWarp.h"    
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;


__global__  void threadsInWarp(threadProperties* threadsDev, int* globalData) {
  
   	
   __shared__ int readtimer [128];
   size_t gid = blockIdx.x *  blockDim.x +  threadIdx.x;
   
   float copyvalue; 
   unsigned long long int startTime = clock64();  
   readtimer[threadIdx.x] = globalData[threadIdx.x];
    
   unsigned long long finishTime = clock64();  

   copyvalue = readtimer[threadIdx.x];

   // Calculate elapsed time
   
      
   unsigned long long GpuTime = finishTime - startTime;
   copyvalue++; 

   threadsDev[gid].value = copyvalue;
   threadsDev[gid].time = GpuTime;
   threadsDev[gid].thread_x = threadIdx.x;   
   
}


