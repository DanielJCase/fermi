//Daniel J. Case
//2/16/17
//Fermi Paradox Project
//Populate stars and create Adjacency matrix


#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

//TO DO
//Consider we have a square lattice of star positions. Each position can be populated or not (1 or 0).
//This grid can be put into a 1D array in row major order.
//Save matrix in 2D format.

//Lattice width
const unsigned int N = 32;

//Fraction of populated sites
const float p = 0.7;

//const int N = 16;
//const int blocksize = 16;

__global__
void hello(char *a, int *b)
{
	a[threadIdx.x] += b[threadIdx.x];
}

__global__ void populateStars(curandState_t* states, unsigned int* d_out, const unsigned int latWidth, const float prob)
{
	int global_x = threadIdx.x + blockIdx.x*blockDim.x;
	int global_y = threadIdx.y + blockIdx.y*blockDim.y;
	int index = global_x + global_y*latWidth;

	if(index >= latWidth*latWidth)
		return;

	float rand_num = curand_uniform(&states[index]);
	if(rand_num > prob){
		d_out[index] = 0;
	}
	else{
		d_out[index] = 1;
	}
}

/* this GPU kernel function is used to initialize the random states */
__global__ void randInit(unsigned int seed, curandState_t* states, const unsigned int latWidth) {

	int global_x = threadIdx.x + blockIdx.x*blockDim.x;
	int global_y = threadIdx.y + blockIdx.y*blockDim.y;
	int index = global_x + global_y*latWidth;

	if(index >= latWidth*latWidth)
			return;

  /* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[index]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, unsigned int* numbers) {
  /* curand works like rand - except that it takes a state as a parameter */
  numbers[blockIdx.x] = curand(&states[blockIdx.x]) % 100;
}


int main()
{
	const dim3 blockSize(32,32,1);
	const dim3 gridSize(N/blockSize.x + 1, N/blockSize.y + 1);

	//Declare array for star positions on host
	unsigned int* h_stars;

	//Declare array for star positions on device
	unsigned int* d_stars;

	//Allocate space on GPU for d_stars
	checkCudaErrors(cudaMalloc(&d_stars, sizeof(unsigned int)*N*N));

	/* CUDA's random number library uses curandState_t to keep track of the seed value
	     we will store a random state for every thread  */
	curandState_t* states;

	/* allocate space on the GPU for the random states */
	checkCudaErrors(cudaMalloc((void**) &states, N * N * sizeof(curandState_t)));

	/* invoke the GPU to initialize all of the random states */
	randInit<<<gridSize, blockSize>>>(5, states, N);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	/* invoke the kernel to get some random numbers */
	populateStars<<<gridSize, blockSize>>>(states, d_stars, N, p);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);

	printf("%s", a);

	cudaMalloc( (void**)&ad, csize );
	cudaMalloc( (void**)&bd, isize );
	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );
/*
	//dim3 dimGrid( 1, 1 );
	//hello<<<dimGrid, dimBlock>>>(ad, bd);
	//cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
	//cudaFree( ad );
	//cudaFree( bd );

	//printf("%s\n", a);
	 *
	 */
	cudaFree(d_stars);
	cudaFree(states);
	return EXIT_SUCCESS;
}
