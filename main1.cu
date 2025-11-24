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


#include "threadsinWarp2D.h"


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
                                      threadsInWarp2D, 0, 0); 
   
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

  threadProperties* threadProp2D; 
  
  int* globalData2D;
  
  
  // pinned data
  
  cudaMallocHost((void**)&threadProp2D ,threadPropertiesDataSize);
  
  cudaMallocHost((void**)&globalData2D ,globalDataSize);

  
  
  for (size_t i = 0; i < BLOCKSIZE; i++) {
      globalData2D[i] = (rand() % 1000) + 1;
  } 
  

  threadProperties* threadProp2DDev; 
  int* globalData2DDev;

  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&threadProp2DDev), threadPropertiesDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&threadProp2DDev), threadPropertiesDataSize)");
    

  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&globalData2DDev), globalDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&globalData2DDev), globalDataSize)");
   
  threads.x = 16;
  threads.y = 8; 
  

  // copy data from host memory to the device:
  status = cudaMemcpy(globalData2DDev, globalData2D, globalDataSize, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status,"cudaMemcpy(globalDataDev, globalData, globalDataSize, cudaMemcpyHostToDevice );");  


  // kernel launch 
  threadsInWarp2D<<< grid, threads >>>(threadProp2DDev, globalData2D);
  status = cudaGetLastError(); 
  // check for cuda errors
  checkCudaErrors( status,"threadsInWarp<<< grid, threads >>>( threadsDev, globalData); ");

  // copy data from device memory to host 
  status = cudaMemcpy(threadProp2D, threadProp2DDev, threadPropertiesDataSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(threads, threadsDev, threadsPropertiesDataSize, cudaMemcpyDeviceToHost) "); 
  
  
  for (int i = 0; i < BLOCKSIZE; i++) {
     
     	  
     printf("Time for execution: %llu clock cycle\n", threadProp2D[i].time ); 
     fprintf (stderr, "threadid.x: %i \n", threadProp2D[i].thread_x);
     fprintf (stderr, "threadid.y: %i \n", threadProp2D[i].thread_y);

  }	  


  return 0;
};	
