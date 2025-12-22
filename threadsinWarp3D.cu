#include <math.h>
#include "threadsinWarp.h"    
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;

__global__  void threadsInWarp3D(threadProperties* threadsDev, int* globalData) {
  
   	
   __shared__ int readtimer [128];
   //size_t gid = blockIdx.x *  (blockDim.x *  blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
   //size_t gid = blockIdx.x *  (blockDim.x *  blockDim.y) + (threadIdx.x * blockDim.y) + threadIdx.y;
 
   size_t gid =  blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + threadIdx.x;
   //size_t gid =  blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.x * blockDim.y) + threadIdx.y;


   int copyvalue; 
   unsigned long long startTime = clock64();  
   readtimer[gid] = globalData[gid];
    
   unsigned long long finishTime = clock64();  

   copyvalue = readtimer[gid];

   // Calculate elapsed time
   
      
   unsigned long long GpuTime = finishTime - startTime;
   copyvalue++; 

   threadsDev[gid].value = copyvalue;
   threadsDev[gid].time = GpuTime;
   threadsDev[gid].thread_x = threadIdx.x;   
   threadsDev[gid].thread_y = threadIdx.y;   
   threadsDev[gid].thread_z = threadIdx.z;   
}


