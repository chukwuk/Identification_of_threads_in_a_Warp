#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <cmath>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>


#include "threadsinWarp3D.h"



#define IDX2C(i,j,ld) (((i)*(ld))+(j))

#define c(x) #x
#define stringify(x) c(x)

#define t(s1,s2) s1##s2
#define tg(s1,s2) t(s1,s2)

#define tgg(s1,s2,s3) tg(tg(s1,s2),s3)
#define tggg(s1,s2,s3,s4) tg(tgg(s1,s2,s3),s4)




using namespace std;


inline
cudaError_t checkCudaErrors(cudaError_t result, string functioncall = "")
{
//#if defined(DEBUG) || defined(_DEBUG)
  //fprintf(stderr, "CUDA Runtime Error: %d\n", result);
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error for this function call ( %s ) : %s\n", 
            functioncall.c_str(), cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
//#endif
  return result;
}


	
int
main( int argc, char* argv[ ] )
{ 
  
  
  int BLOCKSIZE;
  int NUMBLOCKS;
  int MINGRIDSIZE;  
  
  cudaOccupancyMaxPotentialBlockSize( &MINGRIDSIZE, &BLOCKSIZE, 
                                      threadsInWarp3D, 0, 0); 
   
  BLOCKSIZE = 128;
  NUMBLOCKS = (BLOCKSIZE+BLOCKSIZE-1)/BLOCKSIZE;
   
    
  // allocate number of threads in a block  
  dim3 threads(BLOCKSIZE, 1, 1 );

  // allocate number of blocks
  dim3 grid(NUMBLOCKS, 1, 1 );
  
   
  cudaError_t status;
  int threadPropertiesDataSize = (sizeof(threadProperties) * BLOCKSIZE);
  int globalDataSize = (sizeof(int) * BLOCKSIZE);
  fprintf (stderr, "Thread Properties Struct Size %i \n", threadPropertiesDataSize);
  fprintf (stderr, "Global Data Size %i \n", globalDataSize);
  

  // testing for 2D

  threadProperties* threadProp3D; 
  
  int* globalData3D;
  
  
  // pinned data
  
  cudaMallocHost((void**)&threadProp3D ,threadPropertiesDataSize);
  
  cudaMallocHost((void**)&globalData3D ,globalDataSize);

  
  
  for (size_t i = 0; i < BLOCKSIZE; i++) {
      globalData3D[i] = (rand() % 1000) + 1;
  } 
  

  threadProperties* threadProp3DDev; 
  int* globalData3DDev;

  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&threadProp3DDev), threadPropertiesDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&threadProp3DDev), threadPropertiesDataSize)");
    

  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&globalData3DDev), globalDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&globalData3DDev), globalDataSize)");
   
  //threads.x = 4;
  //threads.y = 4; 
  //threads.z = 8; 
  
  
  threads.x = 8;
  threads.y = 8; 
  threads.z = 2; 

  // copy data from host memory to the device:
  status = cudaMemcpy(globalData3DDev, globalData3D, globalDataSize, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status,"cudaMemcpy(globalDataDev, globalData, globalDataSize, cudaMemcpyHostToDevice );");  


  // kernel launch 
  threadsInWarp3D<<< grid, threads >>>(threadProp3DDev, globalData3D);
  status = cudaGetLastError(); 
  // check for cuda errors
  checkCudaErrors( status,"threadsInWarp<<< grid, threads >>>( threadsDev, globalData); ");

  // copy data from device memory to host 
  status = cudaMemcpy(threadProp3D, threadProp3DDev, threadPropertiesDataSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(threadProp3D, threadsProp3DDev, threadsPropertiesDataSize, cudaMemcpyDeviceToHost) "); 
  
  
  for (int i = 0; i < BLOCKSIZE; i++) {
      
     printf("(threadId.x: %i, threadId.y: %i, threadId.z: %i) execution time for copying data from GMEM to SMEM: %llu clock cycle\n", threadProp3D[i].thread_x, threadProp3D[i].thread_y, threadProp3D[i].thread_z, threadProp3D[i].time ); 
     //printf("Time for execution: %llu clock cycle\n", threadProp3D[i].time ); 
     //fprintf (stderr, "threadid.x: %i \n", threadProp3D[i].thread_x);
     //fprintf (stderr, "threadid.y: %i \n", threadProp3D[i].thread_y);
     //fprintf (stderr, "threadid.z: %i \n", threadProp3D[i].thread_z);

  }	  


  return 0;
};	
