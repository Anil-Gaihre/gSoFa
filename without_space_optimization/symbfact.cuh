//#include "graph.h"
#include "wtime.h"
#include "barrier.cuh"
#include <omp.h>
#include <cuda.h>
#include <math.h> 
#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <string>
// #include <math.h>
 
#define _M_LN2  0.693147180559945309417 // Natural log of 2
#define log_2(x) (log(x)/_M_LN2)

using namespace std;
typedef int int_t;
typedef unsigned int uint_t;
typedef unsigned long long ull_t;
// #define test 1
//  #define profile_frontier_sizes 1
// #define maintain_minimum_max_id_fills 1

#define MAX_VAL UINT_MAX //4294967295
#define num_t 24
#define MIN_VAL 0
//#define num_blocks 128
#define num_blocks_cat0 128
#define block_size_cat0 128
#define num_blocks_cat1 128
#define block_size_cat1 128
//#define block_size 128
#define chunk_size_0 128
#define chunk_size_1 128
#define frontier_multiple 0.5
#define debug_illegal_memory_higher_nodes 1
// #define debug_small_concurrent_source 1

// #define profile_edge_check 1
// #define N_src_group 4096
// #define N_src_group 128
// #define overwrite_kernel_config 1


////////////-----------------------//////////////
//////////// Only 1 of the following two variables should be defined. Not both///////////////////////////////
#define enable_fillins_filter_FQ_No_Max_id_update 1
// #define enable_fillins_filter_FQ 1 //This optimization performs better than the default
////////////----------------------/////////////////

//#define N_blocks_source_cat2 4



__host__ __device__ __forceinline__ void swap_ptr_index_vol(volatile int_t* &a, volatile int_t* &b){
    volatile int_t* temp = a;
    a = b;
    b = temp;

}


__host__ __device__ __forceinline__ void swap_ptr_index(int_t* &a, int_t* &b){
    int_t* temp = a;
    a = b;
    b = temp;

}

__device__ __forceinline__ int_t Minimum(int_t a, int_t b)
{
    if (a<b) return a;
    else return b;
}

__host__  __forceinline__ int_t Maximum3(int_t a,int_t b,int_t c)
{
    if (a >b) 
    {
        if (a > c) return a;
        else return c;
    }
    else
    {
        if (b > c) return b;
        else return c;
    }
    //    return (a < b)? b:a;
}

__host__ __device__ __forceinline__ int_t Maximum(uint_t a, uint_t b)
{
    return (a < b)? b:a;
}

__device__ __forceinline__ void  syncgroup(int_t group_id,int_t N_blocks_source)
{
    //sync all the "N_blocks_source" falling into group "group_id"
    return;
}



__device__ __forceinline__ void sync_X_block(int group_id, int N_blocks_source,int* d_lock, int N_groups) 
{
    volatile int *lock = d_lock;    

    // Threadfence and syncthreads ensure global writes 
    // thread-0 reports in with its sync counter
    __threadfence();
    __syncthreads();
    //                int group_bid= blockIdx.x & (N_blocks_source-1);//block id in the group
    int group_bid = blockIdx.x % N_blocks_source;
    int block_offset=group_id*N_blocks_source;
    if (group_bid== 0)//First block in the group
    {
        // Report in ourselves
        if (threadIdx.x == 0)
            lock[group_bid+block_offset] = 1;

        __syncthreads();

        // Wait for everyone else to report in
        //NOTE: change for more than 4 blocks
        int stop_block;
        if(group_id==N_groups-1)
        {
            stop_block=gridDim.x;
        }
        else
        {
            stop_block=block_offset+ N_blocks_source;
        }
        for (int peer_block = block_offset+threadIdx.x; 
                peer_block < stop_block; peer_block += blockDim.x)
            while (ThreadLoad(d_lock + peer_block) == 0)
                __threadfence_block();

        __syncthreads();

        // Let everyone know it's safe to proceed
        for (int peer_block = block_offset+threadIdx.x; 
                peer_block < stop_block; peer_block += blockDim.x)
            lock[peer_block] = 0;
    }
    else
    {
        if (threadIdx.x == 0)
        {
            // Report in
            // lock[blockIdx.x] = 1;
            lock[group_bid+block_offset] = 1;


            // Wait for acknowledgment
            //                         while (ThreadLoad (d_lock + blockIdx.x) != 0)
            while (ThreadLoad (d_lock + group_bid+block_offset) != 0)
                __threadfence_block();
            //  while (ThreadLoad (d_lock + group_bid+block_offset) == 1)
            //      __threadfence_block();
        }
        __syncthreads();
    }
}

__device__ __forceinline__ int_t validated(int_t* original_cost,int_t max_id_offset,int_t vert_count)
{
    // if (original_cost < max_id_offset)

    //original_cost=max_id_offset+vert_count; // Initializing the max_id. This is relatively small in number as the traversal
    int_t temp= atomicCAS(original_cost, (*original_cost > max_id_offset) ? (max_id_offset + vert_count) : *original_cost, (max_id_offset+vert_count));
    // from a source doesn't reach to all the vertices
    //Customized atomic operation needs to be developed insetad if atomicMin() in order to avoid this initialization
    //Also after symmetric pruning this visit will further decrease.

    return temp;
}



__global__ void Compute_fillins_merge_traverse_multiGPU(uint_t* cost_array_d,int_t* fill_in_d, int_t* frontier,
        int_t* next_frontier,int_t vert_count,int_t* csr, int_t* col_st,int_t* col_ed, ull_t* fill_count,int_t gpu_id,int_t N_gpu,
        int_t* src_frontier_d, int_t* next_src_frontier_d,int_t* source_d, int_t* frontier_size, int_t* next_frontier_size, 
        int_t* lock_d,int_t N_groups,  unsigned long long* atomic_time_all_threads,unsigned int* total_frontier_size,
        ull_t* total_edge_check,int_t* next_front_d, unsigned long long* max_id_clock_cycle,unsigned long long* symb_fact_clock_cycle,int_t N_src_group,ull_t* max_id_fill_check_d)

{
    int_t group=0;
    ull_t N_frontier_degrees=0;
    int_t original_thid;
    unsigned long long  Max_id_initialize_cycles=0;
    unsigned long long  symbolic_factorization_cycles=0;
    unsigned long long  symbolic_factorization_cycles_temp=clock64();

    //   #ifdef test
    //   int_t level=0;
    //#endif
    // if (threadIdx.x==0) printf("N_groups: %d\n",N_groups);
    uint_t max_id_offset = MAX_VAL-vert_count;//vert_count*group;
    uint_t group_MAX_VAL = max_id_offset + vert_count;
    int_t reinitialize=1;
    int_t temp_GPU_id = gpu_id;
    while (group < N_groups)
    {
        // printf("Started group:%d\n",group);
        int_t level=0; //test for average frontier size
        //printf("N_groups: %d\n",N_groups);
        //Assign sources to the source_d array //128 sources
        int_t thid= blockDim.x*blockIdx.x+threadIdx.x;
        original_thid=thid; 
        // if (original_thid == 0 && gpu_id == 0) printf("My temp GPU ID:%d  Original GPUid:%d\n",temp_GPU_id,gpu_id);       
        // printf("N_src_group: %d\n",N_src_group);
        while (thid < N_src_group)
        {
            // source_d[thid]=temp_GPU_id + thid * N_gpu + 1 + group * N_src_group;
            source_d[thid]=gpu_id + thid * N_gpu + 1 + group * N_src_group;
            // source_d[thid]=group*N_src_group+ thid+1;
            //  printf("source: %d\n", source_d[thid]); //Works
            thid+=(blockDim.x*gridDim.x);
        }
        //global_barrier.sync_grid_opt();
        sync_X_block(0,gridDim.x,lock_d,1);
        //  if (original_thid==0) printf("Working on group: %d\n",group);


        //Initialize all the cost of the vertices to be 0
        unsigned long long  Max_id_initialize_cycles_temp=clock64();
        if (reinitialize==1)
        {
            for (thid=original_thid; thid < (vert_count*N_src_group); thid+=(blockDim.x*gridDim.x))
            {
                //if (reinitialize==1)
                cost_array_d[thid]=group_MAX_VAL;
                fill_in_d[thid]=0; // For bitwise flag per vertex
            }
            reinitialize=0;
        }
        //global_barrier.sync_grid_opt();
        sync_X_block(0,gridDim.x,lock_d,1);
        // Max_id_initialize_cycles+=(clock64()-Max_id_initialize_cycles_temp);
        //Assign cost of the sources as min val

        for (thid=original_thid; thid < N_src_group;thid+=(blockDim.x*gridDim.x))
        {
            int_t cost_array_offset=thid*vert_count;//int_t cost_array_offset=source_id*vert_count;
            //cost_array_d[cost_array_offset+source_d[thid]]=MIN_VAL;
#ifdef debug_illegal_memory_higher_nodes
            if (source_d[thid] < vert_count)
#endif

                cost_array_d[cost_array_offset+source_d[thid]]=max_id_offset; //max_id_offset=MIN_VAL for the current group
        }

        // if (source_d[blockIdx.x] < vert_count)
        // {
        for (int_t src_id=blockIdx.x ; src_id < N_src_group;src_id +=gridDim.x)
            //for (int_t src_id=blockIdx.x ; src_id < N_src_group;src_id +=blockDim.x)
        {
            if (source_d[src_id] < vert_count)
            {
                for (int_t b_tid=threadIdx.x+col_st[source_d[src_id]]; b_tid < col_ed[source_d[src_id]]; b_tid+=blockDim.x)
                {
                    int_t neighbor=csr[b_tid];
#ifdef profile_edge_check
                    total_edge_check[original_thid] ++;
#endif
                    int_t cost_array_offset=src_id*vert_count;//int_t cost_array_offset=source_id*vert_count;
                    // cost_array_d[cost_array_offset+ neighbor]=MIN_VAL;//Initialising all the neighbots costs with minimum cost
                    cost_array_d[cost_array_offset+ neighbor]=max_id_offset;// //max_id_offset=MIN_VAL for the current group

                    if (neighbor >= source_d[src_id]) continue;
                    // start_time_atomic=clock64();
                    int_t front_position=atomicAdd(frontier_size,1);
                    // time_atomic+=(clock64()-start_time_atomic);
                    frontier[front_position]=neighbor;
                    src_frontier_d[front_position]=src_id;//save the source position in the array not the source itself
                }
            }
        }
        // }

        sync_X_block(0,gridDim.x,lock_d,1);

        while(frontier_size[0]!=0)
        {
            //   printf("Frontier size: %d\n",frontier_size[0]);
            for(thid=original_thid;thid< frontier_size[0];/*thid+=(gridDim.x*blockDim.x)*/)
            {
                int_t front=frontier[thid];
                N_frontier_degrees += (col_ed[front]-col_st[front]);

                int_t cost_array_offset=src_frontier_d[thid] * vert_count;

                if (src_frontier_d[thid] > 0) printf("Found #src > 1\n");
                if (cost_array_offset!=0) printf("condition for more than  1 src\n");
                //  int_t cost = Maximum(front,cost_array_d[cost_array_offset+front]);
                uint_t cost = Maximum(front+max_id_offset,cost_array_d[cost_array_offset+front]);
                int_t src=source_d[src_frontier_d[thid]];
                if (src < vert_count)
                {
                    for (int k=col_st[front]; k<col_ed[front];k++)
                    {
                        int_t m=csr[k];
#ifdef profile_edge_check
                        total_edge_check[original_thid] ++;
#endif

                        if (cost_array_d[cost_array_offset+m] > cost)
                        {
#ifdef enable_fillins_filter_FQ_No_Max_id_update
                            max_id_fill_check_d[original_thid]++;
                            //     if (atomicMax(&fill_in_d[cost_array_offset+m],src) < src)
                            if  (fill_in_d[cost_array_offset+m] < src )

#endif
                            {
                                max_id_fill_check_d[original_thid]++;
                                if (atomicMin(&cost_array_d[cost_array_offset+m],cost) > cost)
                                {

                                    //  printf("detected high cost!\n");                              
                                    if ((m + max_id_offset) > cost)
                                    {

                                        max_id_fill_check_d[original_thid]++;
                                        if (atomicMax(&fill_in_d[cost_array_offset+m],src) < src)
   
                                        {
                                            //Condition telling that the fill-in is not yet detected
                                            atomicAdd(fill_count,1);
                                        }

#ifndef enable_fillins_filter_FQ_No_Max_id_update

                                        else
                                        {
                                            continue; //Don't put the fill-in into the frontier queue because its neighbors already have <= the max_id
                                            // that this fill-in proposes
                                        }
#endif
                                        //  time_atomic+=(clock64()-start_time_atomic);
                                    }

                                    if (m < src)
                                    {
                                        //  start_time_atomic=clock64();
                                        int_t front_position=atomicAdd(next_frontier_size,1);
                                        //  time_atomic+=(clock64()-start_time_atomic);
                                        next_frontier[front_position]=m;
                                        next_src_frontier_d[front_position]=src_frontier_d[thid];
                                    }
                                    //  if (m > cost)

                                }
                            }
                        }
                    }
                }
                thid=atomicAdd(next_front_d,1);

            }

            //global_barrier.sync_grid_opt();
            sync_X_block(0,gridDim.x,lock_d,1);
            swap_ptr_index(frontier,next_frontier);
            swap_ptr_index(src_frontier_d,next_src_frontier_d);
            if (original_thid==0)
            {
#ifdef profile_frontier_sizes
                total_frontier_size[group]+=frontier_size[0];
#endif

                frontier_size[0]=next_frontier_size[0];
                //        printf("next_frontier_size: %d\n",next_frontier_size[0]);
                next_frontier_size[0]=0;
                next_front_d[0]=blockDim.x*gridDim.x;
                //        printf("Finished level:%d \n", level);
            }
#ifdef profile_frontier_sizes
            level++;
#endif

            //global_barrier.sync_grid_opt();
            sync_X_block(0,gridDim.x,lock_d,1);

        }
#ifdef profile_frontier_sizes
        if (original_thid==0)
        {
            // if (group==0) printf("total_frontier_size[group]:%d\n",total_frontier_size[0]);
            // if (group==0) printf("level:%d\n",level);
            if (level!=0) total_frontier_size[group]/=level; //averaging the frontier sizes 
            //   if (group==0) printf("total_frontier_size[group]:%d\n",total_frontier_size[0]);
        }
#endif

        group+=N_gpu;
        max_id_offset-=vert_count;
        if ((max_id_offset-vert_count) < 0)
        {
            max_id_offset=MAX_VAL-vert_count;           
            reinitialize=1;
        }
        group_MAX_VAL = max_id_offset + vert_count;
        temp_GPU_id = N_gpu -1- temp_GPU_id; //Scalar Optimization of workload balancing on scaling
    }

    //  total_edge_check[original_thid] = N_frontier_degrees;
    //  symbolic_factorization_cycles+=(clock64()-symbolic_factorization_cycles_temp);
    //  atomic_time_all_threads[blockDim.x*blockIdx.x+threadIdx.x]=time_atomic;
    //  max_id_clock_cycle [blockDim.x*blockIdx.x+threadIdx.x] = Max_id_initialize_cycles;
    //   symb_fact_clock_cycle [blockDim.x*blockIdx.x+threadIdx.x] = symbolic_factorization_cycles;
}


__global__ void Compute_fillins_merge_traverse_multiGPU_warp_centric_original(uint_t* cost_array_d,int_t* fill_in_d, int_t* frontier,
    int_t* next_frontier,int_t vert_count,int_t* csr, int_t* col_st,int_t* col_ed, ull_t* fill_count,int_t gpu_id,int_t N_gpu,
    int_t* src_frontier_d, int_t* next_src_frontier_d,int_t* source_d, int_t* frontier_size, int_t* next_frontier_size, 
    int_t* lock_d,int_t N_groups,  unsigned long long* atomic_time_all_threads,unsigned int* total_frontier_size,
    ull_t* total_edge_check,int_t* next_front_d, unsigned long long* max_id_clock_cycle,unsigned long long* symb_fact_clock_cycle,int_t N_src_group,
    ull_t* max_id_fill_check_d,int_t NWarps)

{
int_t group=0;
ull_t N_frontier_degrees=0;
int_t original_thid;
unsigned long long  Max_id_initialize_cycles=0;
unsigned long long  symbolic_factorization_cycles=0;
unsigned long long  symbolic_factorization_cycles_temp=clock64();


//   #ifdef test
//   int_t level=0;
//#endif
// if (threadIdx.x==0) printf("N_groups: %d\n",N_groups);
uint_t max_id_offset = MAX_VAL-vert_count;//vert_count*group;
uint_t group_MAX_VAL = max_id_offset + vert_count;
int_t reinitialize=1;
int_t temp_GPU_id = gpu_id;
// int_t NWarps = ((blockDim.x*gridDim.x) >> 5);
while (group < N_groups)
{
    // printf("Started group:%d\n",group);
    int_t level=0; //test for average frontier size
    //printf("N_groups: %d\n",N_groups);
    //Assign sources to the source_d array //128 sources
    int_t thid= blockDim.x*blockIdx.x+threadIdx.x;
    if (thid ==0) 
    {
        if (group%128==0)
        {
        printf("working on group: %d\n",group/128);
        printf("#Fill-ins till now: %d\n",fill_count[0]);
        }
    }
    original_thid=thid; 
    int_t laneID = threadIdx.x & 0x1f;
    int_t warpId = thid >> 5;
    // if (original_thid == 0 && gpu_id == 0) printf("My temp GPU ID:%d  Original GPUid:%d\n",temp_GPU_id,gpu_id);       
    // printf("N_src_group: %d\n",N_src_group);
    while (thid < N_src_group)
    {
        // source_d[thid]=temp_GPU_id + thid * N_gpu + 1 + group * N_src_group;
        source_d[thid]=gpu_id + thid * N_gpu + 1 + group * N_src_group;
        // source_d[thid]=group*N_src_group+ thid+1;
        //  printf("source: %d\n", source_d[thid]); //Works
        thid+=(blockDim.x*gridDim.x);
    }
    //global_barrier.sync_grid_opt();
    sync_X_block(0,gridDim.x,lock_d,1);
    //  if (original_thid==0) printf("Working on group: %d\n",group);


    //Initialize all the cost of the vertices to be 0
    unsigned long long  Max_id_initialize_cycles_temp=clock64();
    if (reinitialize==1)
    {
        for (thid=original_thid; thid < (vert_count*N_src_group); thid+=(blockDim.x*gridDim.x))
        {
            //if (reinitialize==1)
            cost_array_d[thid]=group_MAX_VAL;
            fill_in_d[thid]=0; // For bitwise flag per vertex
        }
        reinitialize=0;
    }
    //global_barrier.sync_grid_opt();
    sync_X_block(0,gridDim.x,lock_d,1);
    // Max_id_initialize_cycles+=(clock64()-Max_id_initialize_cycles_temp);
    //Assign cost of the sources as min val

    for (thid=original_thid; thid < N_src_group;thid+=(blockDim.x*gridDim.x))
    {
        int_t cost_array_offset=thid*vert_count;//int_t cost_array_offset=source_id*vert_count;
        //cost_array_d[cost_array_offset+source_d[thid]]=MIN_VAL;
#ifdef debug_illegal_memory_higher_nodes
        if (source_d[thid] < vert_count)
#endif

            cost_array_d[cost_array_offset+source_d[thid]]=max_id_offset; //max_id_offset=MIN_VAL for the current group
    }

    // if (source_d[blockIdx.x] < vert_count)
    // {
    for (int_t src_id=blockIdx.x ; src_id < N_src_group;src_id +=gridDim.x)
        //for (int_t src_id=blockIdx.x ; src_id < N_src_group;src_id +=blockDim.x)
    {
        if (source_d[src_id] < vert_count)
        {
            for (int_t b_tid=threadIdx.x+col_st[source_d[src_id]]; b_tid < col_ed[source_d[src_id]]; b_tid+=blockDim.x)
            {
                int_t neighbor=csr[b_tid];
#ifdef profile_edge_check
                total_edge_check[original_thid] ++;
#endif
                int_t cost_array_offset=src_id*vert_count;//int_t cost_array_offset=source_id*vert_count;
                // cost_array_d[cost_array_offset+ neighbor]=MIN_VAL;//Initialising all the neighbots costs with minimum cost
                cost_array_d[cost_array_offset+ neighbor]=max_id_offset;// //max_id_offset=MIN_VAL for the current group

                if (neighbor >= source_d[src_id]) continue;
                // start_time_atomic=clock64();
                int_t front_position=atomicAdd(frontier_size,1);
                // time_atomic+=(clock64()-start_time_atomic);
                frontier[front_position]=neighbor;
                src_frontier_d[front_position]=src_id;//save the source position in the array not the source itself
            }
        }
    }
    // }

    sync_X_block(0,gridDim.x,lock_d,1);

    while(frontier_size[0]!=0)
    {
        // for(int_t thid = original_thid; thid < frontier_size[0];/*thid+=(gridDim.x*blockDim.x)*/)
        for(int_t thGroupId = warpId; thGroupId < frontier_size[0];/*thid+=(gridDim.x*blockDim.x)*/)
        {
            // int_t front=frontier[thid];
            int_t front = frontier[thGroupId];

            // int_t cost_array_offset=src_frontier_d[thid] * vert_count;
            // int_t src_index = src_frontier_d[thid];
            int_t src_index = src_frontier_d[thGroupId];
            int_t cost_array_offset = src_index * vert_count;
            uint_t cost = Maximum(front+max_id_offset,cost_array_d[cost_array_offset+front]);
            int_t src=source_d[src_index];
            // int_t src=source_d[src_frontier_d[thid]];

            if (src < vert_count)
            {
                //    for (int k=col_st[front]; k<col_ed[front];k++)
                // if (group==50)
                // {
                //     int a=5;
                // }
                for (int k = col_st[front] + laneID; k < col_ed[front]; k += 32)//for 1 warp of threads
                
                {
                    int_t m=csr[k];
#ifdef profile_edge_check
                    total_edge_check[original_thid] ++;
#endif

                    if (cost_array_d[cost_array_offset+m] > cost)

                    {
#ifdef enable_fillins_filter_FQ_No_Max_id_update
                        // max_id_fill_check_d[original_thid]++;
                            // if (atomicMax(&fill_in_d[cost_array_offset+m],src) < src)
                        if  (fill_in_d[cost_array_offset+m] < src )
#endif
                        {
                            // max_id_fill_check_d[original_thid]++;
                            if (atomicMin(&cost_array_d[cost_array_offset+m],cost) > cost)
                            {

                                //  printf("detected high cost!\n");                              
                                if ((m + max_id_offset) > cost)
                                {
                                    // max_id_fill_check_d[original_thid]++;
                                    if (atomicMax(&fill_in_d[cost_array_offset+m],src) < src)
                                    {
                                        //Condition telling that the fill-in is not yet detected
                                        atomicAdd(fill_count,1);
                                    }

#ifndef enable_fillins_filter_FQ_No_Max_id_update

                                    // else
                                    // {
                                    //     continue; //Don't put the fill-in into the frontier queue because its neighbors already have <= the max_id
                                    //     // that this fill-in proposes
                                    // }
#endif
                                    //  time_atomic+=(clock64()-start_time_atomic);
                                }

                                if (m < src)
                                {
                                    //  start_time_atomic=clock64();
                                    int_t front_position=atomicAdd(next_frontier_size,1);
                                    //  time_atomic+=(clock64()-start_time_atomic);
                                    next_frontier[front_position]=m;
                                    // next_src_frontier_d[front_position]=src_frontier_d[thid];
                                    next_src_frontier_d[front_position]=src_index;
                                }
                                //  if (m > cost)

                            }
                        }
                    }
                }
            }
            if (laneID==0) thGroupId = atomicAdd(next_front_d,1);
         
           thGroupId= __shfl_sync(0xffffffff, thGroupId, 0);  
         
            // thid = atomicAdd(next_front_d,1);

        }

        //global_barrier.sync_grid_opt();
        sync_X_block(0,gridDim.x,lock_d,1);
        swap_ptr_index(frontier,next_frontier);
        swap_ptr_index(src_frontier_d,next_src_frontier_d);
        if (original_thid==0)
        {
#ifdef profile_frontier_sizes
            total_frontier_size[group]+=frontier_size[0];
#endif

            frontier_size[0]=next_frontier_size[0];
            //        printf("next_frontier_size: %d\n",next_frontier_size[0]);
            next_frontier_size[0]=0;
            // next_front_d[0]=blockDim.x*gridDim.x;
            next_front_d[0]=NWarps;
            //        printf("Finished level:%d \n", level);
        }
#ifdef profile_frontier_sizes
        level++;
#endif

        //global_barrier.sync_grid_opt();
        sync_X_block(0,gridDim.x,lock_d,1);

    }
#ifdef profile_frontier_sizes
    if (original_thid==0)
    {
        // if (group==0) printf("total_frontier_size[group]:%d\n",total_frontier_size[0]);
        // if (group==0) printf("level:%d\n",level);
        if (level!=0) total_frontier_size[group]/=level; //averaging the frontier sizes 
        //   if (group==0) printf("total_frontier_size[group]:%d\n",total_frontier_size[0]);
    }
#endif

    group+=N_gpu;
    max_id_offset-=vert_count;
    if ((max_id_offset-vert_count) < 0)
    {
        max_id_offset=MAX_VAL-vert_count;           
        reinitialize=1;
    }
    group_MAX_VAL = max_id_offset + vert_count;
    temp_GPU_id = N_gpu -1- temp_GPU_id; //Scalar Optimization of workload balancing on scaling
}

//  total_edge_check[original_thid] = N_frontier_degrees;
//  symbolic_factorization_cycles+=(clock64()-symbolic_factorization_cycles_temp);
//  atomic_time_all_threads[blockDim.x*blockIdx.x+threadIdx.x]=time_atomic;
//  max_id_clock_cycle [blockDim.x*blockIdx.x+threadIdx.x] = Max_id_initialize_cycles;
//   symb_fact_clock_cycle [blockDim.x*blockIdx.x+threadIdx.x] = symbolic_factorization_cycles;
}


__global__ void Compute_fillins_merge_traverse_multiGPU_warp_centric(uint_t* cost_array_d,int_t* fill_in_d, int_t* frontier,
        int_t* next_frontier,int_t vert_count,int_t* csr, int_t* col_st,int_t* col_ed, ull_t* fill_count,int_t gpu_id,int_t N_gpu,
        int_t* src_frontier_d, int_t* next_src_frontier_d,int_t* source_d, int_t* frontier_size, int_t* next_frontier_size, 
        int_t* lock_d,int_t N_groups,  unsigned long long* atomic_time_all_threads,unsigned int* total_frontier_size,
        ull_t* total_edge_check,int_t* next_front_d, unsigned long long* max_id_clock_cycle,unsigned long long* symb_fact_clock_cycle,int_t N_src_group,
        ull_t* max_id_fill_check_d,int_t NWarps)

{
    int_t group=0;
    // group = 388135;
    ull_t N_frontier_degrees=0;
    int_t original_thid;
    unsigned long long  Max_id_initialize_cycles=0;
    unsigned long long  symbolic_factorization_cycles=0;
    unsigned long long  symbolic_factorization_cycles_temp=clock64();
    int_t N_no_init = (4294967295/vert_count); //(2^32-1/|V|)
    int_t no_init_cnt=0;

    //   #ifdef test
    //   int_t level=0;
    //#endif
    // if (threadIdx.x==0) printf("N_groups: %d\n",N_groups);
    uint_t max_id_offset = MAX_VAL-vert_count;//vert_count*group;
    // uint_t group_MAX_VAL = max_id_offset + vert_count;
    uint_t group_MAX_VAL = MAX_VAL;
    int_t reinitialize=1;
    int_t temp_GPU_id = gpu_id;
    // int_t NWarps = ((blockDim.x*gridDim.x) >> 5);
    while (group < N_groups)
     
    {
        // printf("Started group:%d\n",group);
        int_t level=0; //test for average frontier size
        //printf("N_groups: %d\n",N_groups);
        //Assign sources to the source_d array //128 sources
        int_t thid= blockDim.x*blockIdx.x+threadIdx.x;
        // if (thid ==0) 
        // {
        //     printf("working on src: %d\n",group);
        //     // if (group%128==0)
        //     // {
        //     // printf("working on group: %d\n",group/128);
        //     printf("#Fill-ins till now: %d\n",fill_count[0]);
        //     // }
        // }
        original_thid=thid; 
        int_t laneID = threadIdx.x & 0x1f;
        int_t warpId = thid >> 5;
        // if (original_thid == 0 && gpu_id == 0) printf("My temp GPU ID:%d  Original GPUid:%d\n",temp_GPU_id,gpu_id);       
        // printf("N_src_group: %d\n",N_src_group);
        while (thid < N_src_group)
        {
            // source_d[thid]=temp_GPU_id + thid * N_gpu + 1 + group * N_src_group;
            source_d[thid]=gpu_id + thid * N_gpu + 1 + group * N_src_group;
            // if (source_d[thid] > vert_count) printf("Illegal source assigned to source_d!\n");
            // source_d[thid]=group*N_src_group+ thid+1;
            //  printf("source: %d\n", source_d[thid]); //Works
            thid+=(blockDim.x*gridDim.x);
        }
        //global_barrier.sync_grid_opt();
        sync_X_block(0,gridDim.x,lock_d,1);
        //  if (original_thid==0) printf("Working on group: %d\n",group);


        //Initialize all the cost of the vertices to be 0
        unsigned long long  Max_id_initialize_cycles_temp=clock64();
        //  if (reinitialize==1)
         if (no_init_cnt==0)//Condition for (re-)initilization
        {
            // if (original_thid==0) printf("Initializing at group: %d\n",group);
            for (thid=original_thid; thid < (vert_count*N_src_group); thid+=(blockDim.x*gridDim.x))
            {
                //if (reinitialize==1)
                cost_array_d[thid]=group_MAX_VAL;
                 fill_in_d[thid]=0; // For bitwise flag per vertex
            }
            // reinitialize=0;
            // no_init_cnt=0;
        }
        no_init_cnt++;
        //global_barrier.sync_grid_opt();
        sync_X_block(0,gridDim.x,lock_d,1);
        // Max_id_initialize_cycles+=(clock64()-Max_id_initialize_cycles_temp);
        //Assign cost of the sources as min val

        for (thid=original_thid; thid < N_src_group;thid+=(blockDim.x*gridDim.x))
        {
            int_t cost_array_offset=thid*vert_count;//int_t cost_array_offset=source_id*vert_count;
            //cost_array_d[cost_array_offset+source_d[thid]]=MIN_VAL;
#ifdef debug_illegal_memory_higher_nodes
            if (source_d[thid] < vert_count)
#endif
{
    #ifdef debug_small_concurrent_source
                if (cost_array_offset!=0) printf("Err!: cost_array_offset!=0\n");
    #endif
                cost_array_d[cost_array_offset+source_d[thid]]=max_id_offset; //max_id_offset=MIN_VAL for the current group
}
        }

        // if (source_d[blockIdx.x] < vert_count)
        // {
        for (int_t src_id=blockIdx.x ; src_id < N_src_group;src_id +=gridDim.x)
            //for (int_t src_id=blockIdx.x ; src_id < N_src_group;src_id +=blockDim.x)
        {
            if (source_d[src_id] < vert_count)
            {
                for (int_t b_tid=threadIdx.x+col_st[source_d[src_id]]; b_tid < col_ed[source_d[src_id]]; b_tid+=blockDim.x)
                {
                    int_t neighbor=csr[b_tid];
#ifdef profile_edge_check
                    total_edge_check[original_thid] ++;
#endif
                    int_t cost_array_offset=src_id*vert_count;//int_t cost_array_offset=source_id*vert_count;
                    // cost_array_d[cost_array_offset+ neighbor]=MIN_VAL;//Initialising all the neighbots costs with minimum cost
                    #ifdef debug_small_concurrent_source
                    if (cost_array_offset!=0) printf("Err!: cost_array_offset!=0\n");
                    #endif
                    cost_array_d[cost_array_offset+ neighbor]=max_id_offset;// //max_id_offset=MIN_VAL for the current group

                    if (neighbor >= source_d[src_id]) continue;
                    // start_time_atomic=clock64();
                    int_t front_position=atomicAdd(frontier_size,1);
                    #ifdef debug_small_concurrent_source
                    if (front_position > vert_count-1) printf("Err!: Allocated frontier overflow\n");
                    #endif
                    // time_atomic+=(clock64()-start_time_atomic);
                    frontier[front_position]=neighbor;
                    src_frontier_d[front_position]=src_id;//save the source position in the array not the source itself
                }
            }
        }
        // }

        sync_X_block(0,gridDim.x,lock_d,1);

        while(frontier_size[0]!=0)
        {
            // for(int_t thid = original_thid; thid < frontier_size[0];/*thid+=(gridDim.x*blockDim.x)*/)
            for(int_t thGroupId = warpId; thGroupId < frontier_size[0];/*thid+=(gridDim.x*blockDim.x)*/)
            {
                // int_t front=frontier[thid];
                int_t front = frontier[thGroupId];

                // int_t cost_array_offset=src_frontier_d[thid] * vert_count;
                // int_t src_index = src_frontier_d[thid];
                int_t src_index = src_frontier_d[thGroupId];
                int_t cost_array_offset = src_index * vert_count;
                #ifdef debug_small_concurrent_source
                if (cost_array_offset!=0) printf("Err!: cost_array_offset!=0\n");
                #endif
                // if (src_frontier_d[thGroupId] > 0) printf("Found #src > 1\n");
                // if (cost_array_offset!=0) printf("condition for more than  1 src\n");

                uint_t cost = Maximum(front+max_id_offset,cost_array_d[cost_array_offset+front]);
                int_t src=source_d[src_index];
                // int_t src=source_d[src_frontier_d[thid]];

                if (src < vert_count)
                {
                    //    for (int k=col_st[front]; k<col_ed[front];k++)
                    // if (group==50)
                    // {
                    //     int a=5;
                    // }
                    for (int k = col_st[front] + laneID; k < col_ed[front]; k += 32)//for 1 warp of threads
                    
                    {
                        int_t m=csr[k];
#ifdef profile_edge_check
                        total_edge_check[original_thid] ++;
#endif

                        if (cost_array_d[cost_array_offset+m] > cost)

                        {
#ifdef enable_fillins_filter_FQ_No_Max_id_update
                            // max_id_fill_check_d[original_thid]++;
                                // if (atomicMax(&fill_in_d[cost_array_offset+m],src) < src)
                            if  (fill_in_d[cost_array_offset+m] < src )
#endif
                            {
                                // max_id_fill_check_d[original_thid]++;
                                if (atomicMin(&cost_array_d[cost_array_offset+m],cost) > cost)
                                {

                                    //  printf("detected high cost!\n");                              
                                    if ((m + max_id_offset) > cost)
                                    {
                                        // max_id_fill_check_d[original_thid]++;
                                        if (atomicMax(&fill_in_d[cost_array_offset+m],src) < src)
                                        {
                                            //Condition telling that the fill-in is not yet detected
                                            atomicAdd(fill_count,1);
                                        }

#ifndef enable_fillins_filter_FQ_No_Max_id_update

                                        // else
                                        // {
                                        //     continue; //Don't put the fill-in into the frontier queue because its neighbors already have <= the max_id
                                        //     // that this fill-in proposes
                                        // }
#endif
                                        //  time_atomic+=(clock64()-start_time_atomic);
                                    }

                                    if (m < src)
                                    {
                                        //  start_time_atomic=clock64();
                                        int_t front_position=atomicAdd(next_frontier_size,1);
                                        #ifdef debug_small_concurrent_source
                                        if (front_position > vert_count-1) printf("Err!: Allocated next_frontier overflow\n");
                                        #endif
                                        //  time_atomic+=(clock64()-start_time_atomic);
                                        next_frontier[front_position]=m;
                                        // next_src_frontier_d[front_position]=src_frontier_d[thid];
                                        next_src_frontier_d[front_position]=src_index;
                                    }
                                    //  if (m > cost)

                                }
                            }
                        }
                    }
                }
                if (laneID==0) thGroupId = atomicAdd(next_front_d,1);
             
               thGroupId= __shfl_sync(0xffffffff, thGroupId, 0);  
             
                // thid = atomicAdd(next_front_d,1);

            }

            //global_barrier.sync_grid_opt();
            sync_X_block(0,gridDim.x,lock_d,1);
            swap_ptr_index(frontier,next_frontier);
            swap_ptr_index(src_frontier_d,next_src_frontier_d);
            if (original_thid==0)
            {
#ifdef profile_frontier_sizes
                total_frontier_size[group]+=frontier_size[0];
#endif

                frontier_size[0]=next_frontier_size[0];
                //        printf("next_frontier_size: %d\n",next_frontier_size[0]);
                next_frontier_size[0]=0;
                // next_front_d[0]=blockDim.x*gridDim.x;
                next_front_d[0]=NWarps;
                //        printf("Finished level:%d \n", level);
            }
#ifdef profile_frontier_sizes
            level++;
#endif

            //global_barrier.sync_grid_opt();
            sync_X_block(0,gridDim.x,lock_d,1);

        }
#ifdef profile_frontier_sizes
        if (original_thid==0)
        {
            // if (group==0) printf("total_frontier_size[group]:%d\n",total_frontier_size[0]);
            // if (group==0) printf("level:%d\n",level);
            if (level!=0) total_frontier_size[group]/=level; //averaging the frontier sizes 
            //   if (group==0) printf("total_frontier_size[group]:%d\n",total_frontier_size[0]);
        }
#endif

        group+=N_gpu;
        // max_id_offset =0;
         max_id_offset-=vert_count;
         if (no_init_cnt >= N_no_init)//All threads must come here. Condition for re-intialization
        //  if ((max_id_offset-vert_count) < 0)
        // if ((max_id_offset-3*vert_count) < vert_count)
        {
            if (original_thid==0) printf("reinitialize group: %d\n",group);
            max_id_offset=MAX_VAL-vert_count;           
            // reinitialize=1;
            no_init_cnt=0;
        }
        group_MAX_VAL = max_id_offset + vert_count;
        // group_MAX_VAL = MAX_VAL;
        temp_GPU_id = N_gpu -1- temp_GPU_id; //Scalar Optimization of workload balancing on scaling
    }

    //  total_edge_check[original_thid] = N_frontier_degrees;
    //  symbolic_factorization_cycles+=(clock64()-symbolic_factorization_cycles_temp);
    //  atomic_time_all_threads[blockDim.x*blockIdx.x+threadIdx.x]=time_atomic;
    //  max_id_clock_cycle [blockDim.x*blockIdx.x+threadIdx.x] = Max_id_initialize_cycles;
    //   symb_fact_clock_cycle [blockDim.x*blockIdx.x+threadIdx.x] = symbolic_factorization_cycles;
}


int Compute_Src_group(int vert_count)
{
    cout<<"Start finding N_src_per_group"<<endl;
    double temp = 2147483648/(double)(6*vert_count);
    if (temp > vert_count)
    {
        temp=(int)log_2(vert_count/(double)2);
        temp=pow(2,temp);

    }
    else
    {
        temp=(int)log_2(temp);
        temp=pow(2,temp);
    }
    cout<<"Finished finding N_src_per_group"<<endl;
    return (int)temp;
}

void symbfact_min_id(int args,char** argv,int myrank,ull_t& fill_count, double& time) 
//int main(int args, char **argv)
{
    std::cout<<"Input: ./exe beg end csr #GPU\n";
    cout<<"Num_GPU can be assigned any number of GPU. As there is no communication among the MPI process in run-time a GPU can be used multiple times."<<endl;
    if(args!=5){std::cout<<"Wrong input\n";exit(1);}

    const char *beg_file=argv[1];
    const char *end_file=argv[2];
    const char *csr_file=argv[3];
    int N_virtualGPU = atoi(argv[4]);
    //int num_process=atoi(argv[4]);
    int chunk_size=1024;//atoi(argv[5]);
    int percent_cat0=0;//atoi(argv[6]);
    int percent_cat2=0;//atoi(argv[7]);
    int  N_blocks_source_cat2 = 1;//atoi(argv[8]);
    int N_GPU_Node=1;//atoi(argv[5]);
    // N_virtualGPU = 1;
    //int N_src_group=atoi(argv[10]);
    double* Time = new double [N_virtualGPU];
    ull_t* Edge_check_GPU = new ull_t [N_virtualGPU];
    for (int i=0;i<N_virtualGPU;i++)
    {
        Edge_check_GPU[i] = 0;
    }
    printf("My rank:%d\n",myrank);
    H_ERR(cudaSetDevice(myrank % N_GPU_Node));
    int device;
    H_ERR(cudaGetDevice(&device));
    cout<<"rank "<<myrank<<"has local GPU:"<<device<<endl;
    cout<<"N_blocks_source_cat2: "<<N_blocks_source_cat2<<endl; 
    FILE* fptr;
    if ((fptr = fopen(csr_file,"r")) == NULL)
    {
        printf("Error! opening csr file");
        // Program exits if the file pointer returns NULL.
        exit(1);
    }
    printf("Reading CSR \n");
    int_t edge_count;
    int_t vert_count;
    fscanf(fptr,"%d", &edge_count);
    printf("Number of edges=%d\n", edge_count);

    fscanf(fptr,"%d", &vert_count);
    printf("Number of vertices=%d\n", vert_count);


    int_t* csr= (int*) malloc (edge_count*sizeof(int_t));
    for (int_t i=0;i<edge_count;i++)
    {
        fscanf(fptr,"%d", &csr[i]);
    }
    fclose(fptr); 
    printf("Reading col_begin \n");
    if ((fptr = fopen(beg_file,"r")) == NULL)
    {
        printf("Error! opening col_beg file");
        // Program exits if the file pointer returns NULL.
        exit(1);
    }
    int_t* col_st= (int*) malloc (vert_count*sizeof(int_t));
    for (int_t i=0;i<edge_count;i++)
    {
        fscanf(fptr,"%d", &col_st[i]);
    }
    fclose(fptr); 
    if ((fptr = fopen(end_file,"r")) == NULL)
    {
        printf("Error! opening col end file");
        // Program exits if the file pointer returns NULL.
        exit(1);
    }
    printf("Reading col_end \n");

    int_t* col_ed= (int*) malloc (vert_count*sizeof(int_t));
    for (int_t i=0;i<edge_count;i++)
    {
        fscanf(fptr,"%d", &col_ed[i]);
    }
    fclose(fptr); 


    //  int_t sources_per_process=vert_count/num_process;
    int_t N_chunks= (int) ceil (vert_count/(float)chunk_size);

    int temp_rank=myrank;
    time=0;
    if (temp_rank==0) cout<<"N_chunks: "<<N_chunks<<endl;

    int* next_source_d;
    H_ERR(cudaMalloc((void**) &next_source_d,sizeof(int_t))); 

    int_t N_section=3;



    int_t* frontier_size_d;
    int N_src_group = Compute_Src_group(vert_count);
    // N_src_group=atoi(argv[10]);
    //  N_src_group=1;
    //  N_src_group=128;
    //  N_src_group=32;
    // N_src_group=1;
    // N_src_group=512;

//Divide the sources equally among the GPUs every
    // N_src_group=atoi(argv[10]);
    cout<<"N_src_group: "<<N_src_group<<endl;
    

    H_ERR(cudaMalloc((void**) &frontier_size_d,sizeof(int_t)*N_src_group)); 
    H_ERR(cudaMemset(frontier_size_d, 0, sizeof(int_t)*N_src_group));
    int_t* next_frontier_size_d;
    H_ERR(cudaMalloc((void**) &next_frontier_size_d,sizeof(int_t)*N_src_group)); 
    H_ERR(cudaMemset(next_frontier_size_d, 0, sizeof(int_t)*N_src_group));

    int_t* next_source_global_d;
    H_ERR(cudaMalloc((void**) &next_source_global_d,sizeof(int_t)*N_src_group)); 

    //    int alloc_size=Maximum3(num_blocks,minGridSize,N_groups);


    uint_t* cost_array_d;
    H_ERR(cudaMalloc((void**) &cost_array_d,sizeof(uint_t)*N_src_group*vert_count)); 

    int_t* fill_in_d;
    H_ERR(cudaMalloc((void**) &fill_in_d,sizeof(int_t)*N_src_group*vert_count)); 
    H_ERR(cudaMemset(fill_in_d, 0, sizeof(int_t)*N_src_group*vert_count));

    int* frontier_d;
    H_ERR(cudaMalloc((void**) &frontier_d,sizeof(int_t)*N_src_group*vert_count*frontier_multiple)); 

    int* next_frontier_d;
    H_ERR(cudaMalloc((void**) &next_frontier_d,sizeof(int_t)*N_src_group*vert_count*frontier_multiple)); 

    ull_t* fill_count_d;
    H_ERR(cudaMalloc((void**) &fill_count_d,sizeof(ull_t))); 

    H_ERR(cudaMemset(fill_count_d, 0, sizeof(ull_t)));

    int_t* csr_d;
    H_ERR(cudaMalloc((void**) &csr_d,sizeof(int_t)*edge_count)); 

    int_t* col_st_d;
    H_ERR(cudaMalloc((void**) &col_st_d,sizeof(int_t)*vert_count)); 


    int_t* col_ed_d;
    H_ERR(cudaMalloc((void**) &col_ed_d,sizeof(int_t)*vert_count)); 

    H_ERR(cudaMemcpy(csr_d,csr,sizeof(int_t)*edge_count,cudaMemcpyHostToDevice));

    H_ERR(cudaMemcpy(col_st_d,col_st,sizeof(int_t)*vert_count,cudaMemcpyHostToDevice));

    H_ERR(cudaMemcpy(col_ed_d,col_ed,sizeof(int_t)*vert_count,cudaMemcpyHostToDevice));


    int* src_frontier_d;
    H_ERR(cudaMalloc((void**) &src_frontier_d,sizeof(int_t)*N_src_group*vert_count*frontier_multiple)); 

    int* next_src_frontier_d;
    H_ERR(cudaMalloc((void**) &next_src_frontier_d,sizeof(int_t)*N_src_group*vert_count*frontier_multiple)); 

    int* source_d;
    H_ERR(cudaMalloc((void**) &source_d,sizeof(int_t)*N_src_group));
    int cfg_blockSize=128;
    int BLKS_NUM,blockSize;
    cudaOccupancyMaxPotentialBlockSize( &BLKS_NUM, &blockSize, Compute_fillins_merge_traverse_multiGPU, 0, 0);
    H_ERR(cudaDeviceSynchronize());
    BLKS_NUM = (blockSize * BLKS_NUM)/cfg_blockSize;
    int exp=log2((float)BLKS_NUM);
    BLKS_NUM=pow(2,exp);
    blockSize = cfg_blockSize;
    // BLKS_NUM=128; //Currently implemented for 128 blocks

    // #ifdef overwrite_kernel_config
    // BLKS_NUM=1;
    // blockSize=128;
    // #endif

    cout<<"Detected GridDim: "<<BLKS_NUM<<endl;

    int* lock_d;
    H_ERR(cudaMalloc((void**) &lock_d,sizeof(int)*BLKS_NUM)); //size of lock_d is num of blocks
    H_ERR(cudaMemset(lock_d, 0, sizeof(int)*BLKS_NUM));
    H_ERR(cudaThreadSynchronize());

    int_t N_groups=(ceil) (vert_count/(float)N_src_group);
    cout<<"N_groups: "<<N_groups<<endl;
    Barrier global_barrier(BLKS_NUM); 
    cout<<"BLKS_NUM: "<<BLKS_NUM<<endl;
    cout<<"blockSize: "<<blockSize<<endl;

    cout<<"Running the merge kernel"<<endl;
    unsigned long long* atomic_time_all_threads_d;
    H_ERR(cudaMalloc((void**) &atomic_time_all_threads_d,sizeof( unsigned long long)*BLKS_NUM*blockSize)); //size of lock_d is num of blocks
    unsigned long long* atomic_time_all_threads= (unsigned long long*) malloc (sizeof(unsigned long long)*BLKS_NUM*blockSize);

    unsigned long long* max_id_clock_cycle_d;
    H_ERR(cudaMalloc((void**) &max_id_clock_cycle_d,sizeof( unsigned long long)*BLKS_NUM*blockSize)); //size of lock_d is num of blocks
    H_ERR(cudaMemset(max_id_clock_cycle_d, 0,sizeof(unsigned long long)*BLKS_NUM*blockSize));
    unsigned long long* max_id_clock_cycle= (unsigned long long*) malloc (sizeof(unsigned long long)*BLKS_NUM*blockSize);

    unsigned long long* symb_fact_clock_cycle_d;
    H_ERR(cudaMalloc((void**) &symb_fact_clock_cycle_d,sizeof( unsigned long long)*BLKS_NUM*blockSize)); //size of lock_d is num of blocks
    H_ERR(cudaMemset(symb_fact_clock_cycle_d, 0,sizeof(unsigned long long)*BLKS_NUM*blockSize));
    unsigned long long* symb_fact_clock_cycle= (unsigned long long*) malloc (sizeof(unsigned long long)*BLKS_NUM*blockSize);




    unsigned int* average_frontier_size_d;
    H_ERR(cudaMalloc((void**) &average_frontier_size_d,sizeof( unsigned int)*N_groups)); //size of lock_d is num of blocks
    H_ERR(cudaMemset(average_frontier_size_d,0,sizeof(unsigned int)*N_groups));
    unsigned int* average_frontier_size= (unsigned int*) malloc (sizeof(unsigned int)*N_groups);

    ull_t* total_edge_check_d;
    H_ERR(cudaMalloc((void**) &total_edge_check_d,sizeof( ull_t)*BLKS_NUM*blockSize)); //size of lock_d is num of blocks
    H_ERR(cudaMemset(total_edge_check_d,0,sizeof(ull_t)*BLKS_NUM*blockSize));
    ull_t* total_edge_check= (ull_t*) malloc (sizeof(ull_t)*BLKS_NUM*blockSize);

    int_t NWarps = ((BLKS_NUM*blockSize) >> 5);
    int_t* next_front_d;
    H_ERR(cudaMalloc((void**) &next_front_d,sizeof( int_t))); //size of lock_d is num of blocks
    // int_t next_front=BLKS_NUM*blockSize;
    int_t next_front=NWarps;
    H_ERR(cudaMemcpy(next_front_d,&next_front,sizeof(int_t),cudaMemcpyHostToDevice));


    ull_t* max_id_fill_check= (ull_t*) malloc (sizeof(ull_t)*BLKS_NUM*blockSize);
    ull_t* max_id_fill_check_d;
    H_ERR(cudaMalloc((void**) &max_id_fill_check_d,sizeof(ull_t)*BLKS_NUM*blockSize)); //size of lock_d is num of blocks
    H_ERR(cudaMemset(max_id_fill_check_d,0,sizeof(ull_t)*BLKS_NUM*blockSize));
    
   
//    int_t GPU_ID=0;
    for (int GPU_ID=0; GPU_ID < N_virtualGPU; GPU_ID++)
    {

        myrank = GPU_ID;
        int_t num_process = N_virtualGPU;

        double start_time=wtime();


        //   double start_time_first=wtime();


        // Compute_fillins_merge_traverse_multiGPU <<<BLKS_NUM,blockSize>>> (cost_array_d,fill_in_d,frontier_d,
        //     next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,myrank,num_process,
        //     src_frontier_d,next_src_frontier_d,source_d, frontier_size_d, next_frontier_size_d,
        //     lock_d,N_groups, atomic_time_all_threads_d,average_frontier_size_d,total_edge_check_d,
        //     next_front_d,max_id_clock_cycle_d,symb_fact_clock_cycle_d,N_src_group,max_id_fill_check_d);    

        // Compute_fillins_merge_traverse_multiGPU_warp_centric_original <<<BLKS_NUM,blockSize>>> (cost_array_d,fill_in_d,frontier_d,
        //         next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,myrank,num_process,
        //         src_frontier_d,next_src_frontier_d,source_d, frontier_size_d, next_frontier_size_d,
        //         lock_d,N_groups, atomic_time_all_threads_d,average_frontier_size_d,total_edge_check_d,
        //         next_front_d,max_id_clock_cycle_d,symb_fact_clock_cycle_d,N_src_group,max_id_fill_check_d,NWarps);  

        Compute_fillins_merge_traverse_multiGPU_warp_centric <<<BLKS_NUM,blockSize>>> (cost_array_d,fill_in_d,frontier_d,
            next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,myrank,num_process,
            src_frontier_d,next_src_frontier_d,source_d, frontier_size_d, next_frontier_size_d,
            lock_d,N_groups, atomic_time_all_threads_d,average_frontier_size_d,total_edge_check_d,
            next_front_d,max_id_clock_cycle_d,symb_fact_clock_cycle_d,N_src_group,max_id_fill_check_d,NWarps);  

        H_ERR(cudaDeviceSynchronize());  
        time =((wtime()-start_time)*1000);
        //Reinitialize variables for another GPU
        H_ERR(cudaMemcpy(next_front_d,&next_front,sizeof(int_t),cudaMemcpyHostToDevice));
        H_ERR(cudaMemset(max_id_fill_check_d,0,sizeof(ull_t)*BLKS_NUM*blockSize));
        Time[GPU_ID] = time;
        // Edge_check_GPU[]
        //Calculating total edge checks
        H_ERR(cudaMemcpy(total_edge_check,total_edge_check_d,sizeof(ull_t)*BLKS_NUM*blockSize,cudaMemcpyDeviceToHost));
        for (int_t i=0;i<BLKS_NUM*blockSize;i++)
        {
            Edge_check_GPU[GPU_ID] += total_edge_check[i];
        }
        H_ERR(cudaMemset(total_edge_check_d,0,sizeof(ull_t)*BLKS_NUM*blockSize));

        cout<<"Time for GPU "<<GPU_ID<<" :"<<Time[GPU_ID]<<" ms"<<endl;
        H_ERR(cudaMemcpy(&fill_count,fill_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
        cout<<"Number of fill-ins detected till now: "<<fill_count<<endl;
    }
    cout<<"merge traversal complete!"<<endl;
    //#ifdef test
    H_ERR(cudaMemcpy(atomic_time_all_threads,atomic_time_all_threads_d,sizeof(unsigned long long)*BLKS_NUM*blockSize,cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(average_frontier_size,average_frontier_size_d,sizeof(unsigned int)*N_groups,cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(total_edge_check,total_edge_check_d,sizeof(ull_t)*BLKS_NUM*blockSize,cudaMemcpyDeviceToHost));
    H_ERR(cudaMemset(lock_d, 0, sizeof(int)*BLKS_NUM));
    H_ERR(cudaMemset(frontier_size_d, 0, sizeof(int_t)*N_src_group));
    H_ERR(cudaMalloc((void**) &next_frontier_size_d,sizeof(int_t)*N_src_group)); 

    H_ERR(cudaMemcpy(max_id_clock_cycle,max_id_clock_cycle_d,sizeof(unsigned long long)*BLKS_NUM*blockSize,cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(symb_fact_clock_cycle,symb_fact_clock_cycle_d,sizeof(unsigned long long)*BLKS_NUM*blockSize,cudaMemcpyDeviceToHost));

    H_ERR(cudaMemcpy(max_id_fill_check,max_id_fill_check_d,sizeof(ull_t)*BLKS_NUM*blockSize,cudaMemcpyDeviceToHost));

    double sum=0;
    ull_t Traversed_edge_count=0;
    ull_t Total_max_id_fill_check = 0;
    for (int_t i=0;i<BLKS_NUM*blockSize;i++)
    {
        Total_max_id_fill_check += max_id_fill_check[i];
        Traversed_edge_count+= total_edge_check[i];

        //        cout<<"Total Edge Check["<<i<<"]:"<<total_edge_check[i]<<endl;
        //       cout<<"Total Edge Check["<<i<<"]:"<<total_edge_check[i]<<endl;
    }
    cout<<"cumulative sum of degree of frotniers: "<<Traversed_edge_count<<endl;

    //    sum/=N_groups;
    cout<<"Average Edge Check of all groups: "<<sum<<endl;
    cout<<"TEPS: "<< Traversed_edge_count/(time/1000) <<endl;
    sum=0;

    unsigned long long total_symb_fact_clock_cycle=0;
    unsigned long long total_max_id_clock_cycle =0;
    for (int_t i=0;i< BLKS_NUM*blockSize;i++)
    {
        total_symb_fact_clock_cycle+=symb_fact_clock_cycle[i];
        total_max_id_clock_cycle+=max_id_clock_cycle[i];
    }
    cout<<"Total max_id initialization time: "<<total_max_id_clock_cycle<<endl;
    cout<<"Total symb_fact time: "<<total_symb_fact_clock_cycle<<endl;
    //#endif

#ifdef profile_frontier_sizes
    // std::fstream frontier_size_file;
    char file_name[512];
    sprintf(file_name, "average_frontier_size_group_V_%d_N_src_group_%d.csv", vert_count,N_src_group);
    cout<<file_name<<endl;
    // FILE *fptr;
    //fptr = fopen(file_name,"w");
    // fptr = fopen("test.csv","w");
    // if(fptr == NULL)
    // {
    //    printf("Error in creating file!");   
    //    exit(1);             
    // }

    // fprintf(fptr,"group_id;average_frontier_size");
    // //frontier_size_file.open(file_name,std::fstream::out);
    // std::ofstream frontier_size_file("program3data.txt");
    // // frontier_size_file.open("test.csv",std::fstream::out);
    // frontier_size_file<<"group_id;"<<"average_frontier_size"<<endl;
    for (int i=0;i<N_groups;i++)
    {
        // frontier_size_file<<i<<";"<<average_frontier_size[i]<<endl;
        // fprintf(fptr,"%d;%d\n",i,average_frontier_size[i]);
        cout<<"aver_fron_size;"<<N_src_group<<";"<<i<<";"<<average_frontier_size[i]<<endl;
    }
    // fclose(fptr);
    // frontier_size_file.close();
#endif
    double minimum_time = Time[0];
    double maximum_time = Time[0];
    int_t GPU_slowest_time =0;
    int_t GPU_fastest_time =0;
    int_t GPU_least_edgeChecks =0;
    int_t GPU_highest_edgeChecks =0;
    ull_t max_edge_checks = Edge_check_GPU[0];
    ull_t min_edge_checks = Edge_check_GPU[0];

    for (int i=1; i < N_virtualGPU; i++)
    {
        if (Time[i] < minimum_time) 
        {
            minimum_time = Time[i];
            GPU_fastest_time = i;
        }
        if (Time[i] > maximum_time)
        {
            maximum_time = Time[i];
            GPU_slowest_time = i;
        } 
        if (Edge_check_GPU[i] > max_edge_checks) 
        {
            max_edge_checks = Edge_check_GPU[i];
            GPU_highest_edgeChecks = i;
        }
        if (Edge_check_GPU[i] < min_edge_checks) 
        {
            min_edge_checks = Edge_check_GPU[i];
            GPU_least_edgeChecks = i;
        }
    }

    H_ERR(cudaMemcpy(&fill_count,fill_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
    cout<<"Number of fill-ins detected: "<<fill_count<<endl;
    cout<<"time for fill-in detection: "<<time<<endl;
    cout<<"Total_number of max_id + fill check: "<<Total_max_id_fill_check<<endl;
    cout<<"GPU:"<<GPU_slowest_time<<" Slowest virtual GPU takes: "<<maximum_time<< " ms"<<endl;
    cout<<"GPU:"<<GPU_fastest_time<<" Fastest virtual GPU takes: "<<minimum_time<< " ms"<<endl;
    cout<<"GPU:"<<GPU_least_edgeChecks<<" gets smallest edgechecks of: "<<min_edge_checks<<endl;
    cout<<"GPU:"<<GPU_highest_edgeChecks<<" gets highest edgechecks of: "<<max_edge_checks<<endl;
    cout<<"N_groups: "<<N_groups<<endl;
    return;
}
