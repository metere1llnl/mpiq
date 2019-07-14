/**************************************************************************************************

    MPI CUDA-awareness/GPUdirect Checker for Open MPI and compatibles (e.g., IBM Spectrum MPI)

    Author: Alfredo Metere
    e-mail: metal@icsi.berkeley.edu

    Copyright (c) 2019 Alfredo Metere.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

***************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <hwloc.h>
#include <hwloc/cudart.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define cuChk(call)                                                                       \
{                                                                                         \
	const cudaError_t cudaError = call;                                               \
	if (cudaError != cudaSuccess)                                                     \
	{                                                                                 \
		printf("\nError %s: %d, ", __FILE__, __LINE__ );                          \
		printf("Code %d, reason %s\n", cudaError, cudaGetErrorString(cudaError)); \
		exit(1);                                                                  \
	}                                                                                 \
}                                                                                         \

int main(int argc, char **argv)
{

#ifndef OPEN_MPI
	printf("This program requires OpenMPI or compatible\n");
	exit(-1);
#endif
	hwloc_topology_t topology;
	hwloc_cpuset_t cpuset;
	// Initial values are set negative for early error detection.
	int cpu = -1; int ii = -1;
	int rank = -1, local_rank = -1, totalRanks = -1;    
	int cuDevice = -1;
        
	rank       = atoi(std::getenv("OMPI_COMM_WORLD_RANK"));
        local_rank = atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
             		                    
	hwloc_topology_init(&topology); // Creates a hwloc topology
	hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_WHOLE_IO); // Gets the topology of the system, including attached devices
	hwloc_topology_load(topology);
                                        	 	            
        int cuDevCount = 0;
        cuChk(cudaGetDeviceCount(&cuDevCount));
        cuChk(cudaSetDevice(local_rank % cuDevCount)); // For each local rank (rank running on each node), select CUDA device number matching the rank number
        cuChk(cudaGetDevice(&cuDevice)); // Get properties of the currently selected GPU
                   
        cpuset = hwloc_bitmap_alloc(); // Select cores in node
        hwloc_cudart_get_device_cpuset(topology, cuDevice, cpuset); // Get the logical processors near the selected GPU

	int match = 0;
        hwloc_bitmap_foreach_begin(ii,cpuset) // Cycle through all logical processors in the cpuset. 
        //                                   *** NOTE: This is a preprocessor MACRO. No terminating semicolon. ***
        if (match == local_rank)
        {
 		cpu = ii;
         	break;
        }
        hwloc_bitmap_foreach_end(); // This is a preprocessor MACRO too, but it needs terminating semicolon.
		                                                         	 	                                                                                            hwloc_bitmap_t onecpu = hwloc_bitmap_alloc();
        hwloc_bitmap_set(onecpu, cpu);
        hwloc_set_cpubind(topology, onecpu, 0);
        hwloc_bitmap_free(onecpu);
        hwloc_bitmap_free(cpuset);
        hwloc_topology_destroy(topology);
        char hostname[MPI_MAX_PROCESSOR_NAME];
        gethostname(hostname, sizeof(hostname));
        cpu = sched_getcpu();

	MPI_Init(&argc, &argv);

	int successFlag = 0;
	
	int cs = MPIX_Query_cuda_support();
	int N = 8;
	int verifyArrays = 0;

	cudaEvent_t start, stop;
	cuChk(cudaEventCreate(&start));
	cuChk(cudaEventCreate(&stop));

	if (argc > 1)
	{
		for (int i = 1; i < argc; i++)
		{
			if (!strcmp(argv[i],"-cs"))
			{
				if (rank == 0) printf("Overriding CUDA-awareness (detected: %d)\n",cs);
				cs = atoi(argv[i+1]);
			}

			if (!strcmp(argv[i],"-n"))
			{
				N = atoi(argv[i+1]);
			}
            if (!strcmp(argv[i],"-v"))
            {
                verifyArrays = 1;
            }
		}
	}
	
	
	
	int *d_a = NULL;
	int *h_a = NULL;
	int *lh_a = NULL;

	char hname[MPI_MAX_PROCESSOR_NAME];
	int hlen = -1;  

	MPI_Get_processor_name(hname, &hlen);

	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

	if (rank == 0) printf("CUDA-awareness: %d\n", cs);

	printf("Rank %d selected GPU %d on hostname %s\n",rank, rank%4, hname);

	if (rank == 0)
	{
		h_a = (int *) calloc(N, sizeof(int));
		
		for (int i = 0; i < N; i++)
		{
			h_a[i] = 666;
		}

		cudaEventRecord(start, 0);
		cuChk(cudaMalloc(&d_a, N*sizeof(int)));
 		cuChk(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice)); 
	
		MPI_Request mpiReq;

		for (int i = 1; i < totalRanks; i++)
		{
			if (cs) // This will only work with CUDA-Aware MPI
			{
				MPI_Isend(d_a, N, MPI_INT, i, 0, MPI_COMM_WORLD, &mpiReq);
			}
			else
			{
				int *lh_a = (int *) calloc(N, sizeof(int));
				cuChk(cudaMemcpy(lh_a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost));

                if (verifyArrays)
                {
    				for (int j = 0; j < N; j++)
	    			{
		    			if (lh_a[j] != 666)
			    		{
				    		if (j < 10)
					    	{
						    	printf("Rank %d, Error: lh_a[%d] = %d\n", rank, j, lh_a[j]);
    						}
	    					else
		    				{
			    				MPI_Finalize();	
				    			exit(-1);
					    	}
    					}
	    			}
			    }
				MPI_Isend(lh_a, N, MPI_INT, i, 0, MPI_COMM_WORLD, &mpiReq);
				MPI_Status mpiStatus;
		        MPI_Wait(&mpiReq, &mpiStatus);
			}
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float gpu_time = 0.0f;

		cuChk(cudaEventElapsedTime(&gpu_time, start, stop));
		printf ("Rank %d, CUDA Time: %.5f ms\n", rank, gpu_time);
		MPI_Status mpiStatus;
	        MPI_Wait(&mpiReq, &mpiStatus);
		successFlag = 1;
	}
	else
	{
		MPI_Request mpiReq;
		cuChk(cudaMalloc(&d_a, N*sizeof(int)));
		lh_a = (int*) calloc(N, sizeof(int));

		cuChk( cudaEventRecord(start, 0) );		

		if (cs)
		{
			MPI_Irecv(d_a, N, MPI_INT, 0, 0, MPI_COMM_WORLD, &mpiReq);
	                MPI_Status mpiStatus;
	                MPI_Wait(&mpiReq, &mpiStatus);
		}
		else
		{
			MPI_Irecv(lh_a, N, MPI_INT, 0, 0, MPI_COMM_WORLD, &mpiReq);
	                MPI_Status mpiStatus;
	                MPI_Wait(&mpiReq, &mpiStatus);

        	    	if (verifyArrays)
		        {
    				for (int i = 0; i < N; i++)
		    		{
		    			if (lh_a[i] != 666)
				    	{
					        if (i < 10)
	                		        {
                        				printf("Rank %d, Error: lh_a[%d] = %d\n", rank, i, lh_a[i]);
		        	                }
                			        else
			                        {
	                			        MPI_Finalize();
			                        	exit(-1);
                        			}
					}
				}
			}
			cuChk(cudaMemcpy(d_a, lh_a, N*sizeof(int), cudaMemcpyHostToDevice));
		}

		h_a = (int *) calloc(N, sizeof(int));
		cuChk(cudaMemcpy(h_a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost));	
		
		cuChk(cudaEventRecord(stop, 0));
		cuChk(cudaEventSynchronize(stop));

		float gpu_time = 0.0f;
		cuChk(cudaEventElapsedTime(&gpu_time, start, stop));
		
        	printf ("Rank %d, CUDA Time: %.5f ms\n",rank, gpu_time);

		for (int i = 0; i < N; i++)
		{
			if (h_a[i] != 666)
			{
				if (i < 10)
				{
					printf("Rank %d, Error: h_a[%d] = %d\n",rank,i,h_a[i]);
				}
				else
				{
					MPI_Finalize();
					exit(-1);
				}
			}
		}
		successFlag = 1;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (successFlag)
	{
		printf("Rank %d, SUCCESS!\n", rank);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cuChk(cudaFree(d_a));
	free(h_a);
	free(lh_a);		

	MPI_Finalize();
	return 0;
}
