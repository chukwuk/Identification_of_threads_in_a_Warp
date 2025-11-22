#include <math.h>
#include "threadsinWarp.h"    
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;

__global__  void threadsInWarp2D(threadProperties* threadsDev, int* globalData) {
  
   	
   __shared__ int readtimer [128];
   size_t gid = blockIdx.x *  (blockDim.x *  blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
   //size_t gid = blockIdx.x *  (blockDim.x *  blockDim.y) + (threadIdx.x * blockDim.y) + threadIdx.y;
 


   float copyvalue; 
   unsigned long long startTime = clock();  
   readtimer[gid] = globalData[gid];
    
   unsigned long long finishTime = clock();  

   copyvalue = readtimer[gid];

   // Calculate elapsed time
   
      
   unsigned long long GpuTime = finishTime - startTime;
   copyvalue++; 

   threadsDev[gid].value = copyvalue;
   threadsDev[gid].time = GpuTime;
   threadsDev[gid].thread_x = threadIdx.x;   
   threadsDev[gid].thread_y = threadIdx.y;   
}


