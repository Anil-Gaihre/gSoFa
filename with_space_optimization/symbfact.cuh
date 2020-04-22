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
#include <stdint.h>
// #include "Compute_fill_ins.cuh" //For byte addressing
// #include <math.h>

#define _M_LN2  0.693147180559945309417 // Natural log of 2
#define log_2(x) (log(x)/_M_LN2)

using namespace std;
typedef int int_t;
typedef unsigned int uint_t;
typedef unsigned long long ull_t;
// #define test 1
// #define profile_frontier_sizes 1
// #define maintain_minimum_max_id_fills 1

#define MAX_VAL UINT_MAX
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

// #define thread_centric 1

// #define enable_debug 1
// #define enable_warp_debug 1

// #define frontier_multiple 0.5
// #define reduce_max_id_space 1
#define late_FQ_size_check 1
#define profile_FQ_check_ovehead 1
#define debug_illegal_memory_higher_nodes 1
// #define portion_gpu_memory 0.2 //For given memory space: if assigned 0.2 and GPU memory is 16 GB. Allocate 3.2 GB (20% of 16 GB) to the 4 major datastructures
// #define total_memory_allocation_factor 0.1 //For reequired memory space: if assigned 0.2 and 6*|V|*#concurrent_source
// #define max_id_fraction 0.9 //max_id_fraction =0.6 means maximum of 60% of the allocated space on GPU can be given to max_id

// #define total_memory_allocation_factor 0.1 //For reequired memory space: if assigned 0.2 and 6*|V|*#concurrent_source
#define max_id_fraction 0.9 //max_id_fraction =0.6 means maximum of 60% of the allocated space on GPU can be given to max_id




// #define total_memory_allocation_factor 0.1 //For reequired memory space: if assigned 0.2 and 6*|V|*#concurrent_source
// #define max_id_fraction 0.9 //max_id_fraction =0.6 means maximum of 60% of the allocated space on GPU can be given to max_id

//  (#concurrent_source computed on the basis of amount of GPU memory allocated (8 GB by default) for 4 datastructures)
// Allocate 20% of 6*|V|*#concurrent_source to the 4 major datastructures


// #define max_id_space_allocation 0.2 //assign 0.2=20% w.r.t. #optimium_concurrent_src*|V|
// #define FQspace_allocation 0.2 //assign 0.2=20% w.r.t. #optimium_concurrent_src*|V|  


// #define max_id_space_allocation 0.1 //assign 0.2=20% w.r.t. #optimium_concurrent_src*|V| 

#define profile_dumping_loading_time 1
#define profile_memcpy_bandwidth 1
#define real_allocation_factor 1.1 //if 1.5 % real frontier size is 1.5 times greater than allocated frontier size
static int maximum_front_size;
// #define allocated_frontier_size 1024
// #define allocated_frontier_size 350
// #define allocated_frontier_size 65536
// #define allocated_frontier_size 80000
// #define allocated_frontier_size 131072 //2*65536
// #define allocated_frontier_size 263612 //4*65536 
// #define allocated_frontier_size 527224 //8*65536 
// #define allocated_frontier_size 1048576 //16*65536 
// #define allocated_frontier_size 524288
// #define allocated_frontier_size 1048576

// #define allocated_frontier_size 306749905 //For N_source: 1024 pre2 dataset allocation (50% worse space complexity possible)
// #define allocated_frontier_size 122699962 //For N_source: 1024 pre2 dataset allocation (20% worse space complexity possible) (0.2*1024*659033/1.1) //Note /1.1 for flexiblility in FQ insertion
// #define allocated_frontier_size 61349981  //For N_source: 1024 pre2 dataset allocation (10% worse space complexity possible) (0.1*1024*659033/1.1) //Note /1.1 for flexiblility in FQ insertion

// #define allocated_frontier_size 613499810 //For N_source: 2048 pre2 dataset allocation (50% worse space complexity possible)
// #define allocated_frontier_size 245399924 //For N_source: 2048 pre2 dataset allocation (20% worse space complexity possible) (0.2*2048*659033/1.1) //Note /1.1 for flexiblility in FQ insertion
// #define allocated_frontier_size 122699962  //For N_source: 2048 pre2 dataset allocation (10% worse space complexity possible) (0.1*2048*659033/1.1) //Note /1.1 for flexiblility in FQ insertion

// #define allocated_frontier_size 84210036 //hamrle3 dataset allocation
// #define allocated_frontier_size 33742489
// #define allocated_frontier_size 42178112

#define overwrite_kernel_config 1
// #define enable_debug 1


////////////-----------------------//////////////
//////////// Only 1 of the following two variables should be defined. Not both///////////////////////////////
// #define enable_fillins_filter_FQ_No_Max_id_update 1
#define enable_fillins_filter_FQ 1 //This optimization performs better than the default
////////////----------------------/////////////////

#define debug_illegal_memory_higher_nodes 1



__host__ __device__ __forceinline__ void swap_ptr_index_vol(volatile int_t* &a, volatile int_t* &b){
    volatile int_t* temp = a;
    a = b;
    b = temp;

}


__host__ __device__ __forceinline__ void swap_ptr_index(uint_t* &a, uint_t* &b){
    uint_t* temp = a;
    a = b;
    b = temp;

}
__host__ __device__ __forceinline__ void  swap_ptr_index_uint_8 (uint8_t* &a, uint8_t* &b){
    uint8_t* temp = a;
    a = b;
    b = temp;
}
__host__ __device__ __forceinline__ void  swap_ptr_index_uint_16 (uint16_t* &a, uint16_t* &b){
    uint16_t* temp = a;
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



// __device__ __forceinline__ void sync_X_block(int group_id, int N_blocks_source,int* d_lock, int N_groups) 
// {
//     volatile int *lock = d_lock;    

//     // Threadfence and syncthreads ensure global writes 
//     // thread-0 reports in with its sync counter
//     __threadfence();
//     __syncthreads();
//     //                int group_bid= blockIdx.x & (N_blocks_source-1);//block id in the group
//     int group_bid = blockIdx.x % N_blocks_source;
//     int block_offset=group_id*N_blocks_source;
//     if (group_bid== 0)//First block in the group
//     {
//         // Report in ourselves
//         if (threadIdx.x == 0)
//             lock[group_bid+block_offset] = 1;

//         __syncthreads();

//         // Wait for everyone else to report in
//         //NOTE: change for more than 4 blocks
//         int stop_block;
//         if(group_id==N_groups-1)
//         {
//             stop_block=gridDim.x;
//         }
//         else
//         {
//             stop_block=block_offset+ N_blocks_source;
//         }
//         for (int peer_block = block_offset+threadIdx.x; 
//                 peer_block < stop_block; peer_block += blockDim.x)
//             while (ThreadLoad(d_lock + peer_block) == 0)
//                 __threadfence_block();

//         __syncthreads();

//         // Let everyone know it's safe to proceed
//         for (int peer_block = block_offset+threadIdx.x; 
//                 peer_block < stop_block; peer_block += blockDim.x)
//             lock[peer_block] = 0;
//     }
//     else
//     {
//         if (threadIdx.x == 0)
//         {
//             // Report in
//             // lock[blockIdx.x] = 1;
//             lock[group_bid+block_offset] = 1;


//             // Wait for acknowledgment
//             //                         while (ThreadLoad (d_lock + blockIdx.x) != 0)
//             while (ThreadLoad (d_lock + group_bid+block_offset) != 0)
//                 __threadfence_block();
//             //  while (ThreadLoad (d_lock + group_bid+block_offset) == 1)
//             //      __threadfence_block();
//         }
//         __syncthreads();
//     }
// }

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



__global__ void Initialise_cost_array(uint_t* cost_array_d,
        ull_t total_initialization,uint_t group_MAX_VAL)
{
    int offset=blockDim.x*gridDim.x;
    // int total_initialization=vert_count*N_src_group;
    for (ull_t thid=blockDim.x*blockIdx.x+threadIdx.x; thid < total_initialization;thid+=offset)
    {

        cost_array_d[thid]=group_MAX_VAL;
    }
}



//for uint_16
__global__ void  Compute_fillins_Auxiliary_level_traversal  (int_t* frontier_size, uint_t* cost_array_d,
    uint_t* fill_in_d,uint_t* frontier,
        uint_t* next_frontier,int_t vert_count,
        int_t* csr, int_t* col_st,  int_t* col_ed,
        ull_t* fill_count,
        uint_t* src_frontier_d,uint_t* next_src_frontier_d,
        int_t* source_d,    int_t* next_frontier_size,
        int_t* lock_d, int_t* dump,int_t* load,
        uint_t max_id_offset, int_t* next_front_d,ull_t passed_allocated_frontier_size,
        int_t* my_current_frontier_d, int_t* current_buffer_m,int_t* next_buffer_m, int_t* offset_kernel,
        int_t* swap_CPU_buffers,int_t* frontierchecked,int_t* swap_GPU_buffers_m,int_t* remaining_frontier_size,
        int_t max_vertex_group,int_t first_source,int_t N_gpu, ull_t allocated_size_max_id,ull_t real_allocation)

{

    //A subset of the original kernel that only finishes the traversal for current group
    //The function is called repeatedly in case of reloding the excess frontiers and 
    //or on writing the excess frontiers

    int_t thid= blockDim.x*blockIdx.x+threadIdx.x;
    int_t laneID = threadIdx.x & 0x1f;
    int_t warpId = thid >> 5;
    int_t original_thid=thid;  
    // printf("Inside auxiliary kernel\n");  
#ifdef enable_debug    
    if (original_thid==0) 
    {
        printf("Inside auxiliary kernel\n");
        // printf("At kernel launch. Aux: next_front_d:%d\n",next_front_d[0]);
    }
#endif

    uint_t dump_local=0;

    // for(thid=my_current_frontier_d[original_thid] ;thid < frontier_size[0];/*thid+=(gridDim.x*blockDim.x)*/)
    #ifdef enable_warp_debug
    printf("AuxNotDumping: original thread:%d my_current_frontier_d[original_thid]: %d\n",original_thid, my_current_frontier_d[original_thid]);
    #endif
    // for(int_t thGroupId = my_current_frontier_d[warpId] ;thGroupId < frontier_size[0];)
    for(int_t thGroupId = my_current_frontier_d[original_thid] ;thGroupId < frontier_size[0];)

    {
        // printf("Thread: %d inside\n",laneID);

        int_t front=frontier[thGroupId];   
        // printf("Inside auxiliary kernel: original_thid:%d thid:%d front:%d  frontier_size:%d next_frontier_size:%d\n",original_thid, thid,front,frontier_size[0],ThreadLoad(next_frontier_size));    
        // int_t cost_array_offset=src_frontier_d[thid] * vert_count;
        int_t s_id=src_frontier_d[thGroupId];
        int_t cost_array_offset=(s_id*(2*(first_source+1)+(s_id-1)*N_gpu))/2;//optimization for removing the intermediate bubbles in the cost array        
        // int_t cost_array_offset=src_frontier_d[thid] * max_vertex_group;
        int_t fill_in_offset=s_id *vert_count;   
        //  int_t cost = Maximum(front,cost_array_d[cost_array_offset+front]);
        uint_t cost = Maximum(front+max_id_offset,cost_array_d[cost_array_offset+front]);
        int_t src=source_d[s_id];
        // printf(" auxiliary kernel: original_thid: %d  thid:%d  front:%d src:%d frontier_size: %d  next_frontier_size:%d\n",original_thid,thid,front,src,frontier_size[0],ThreadLoad(next_frontier_size));
        if (src < vert_count)
        {          
            // #ifdef early_exit 
            int degree=  col_ed[front]- col_st[front];       


            if (laneID==0)
            {
                if (degree > (passed_allocated_frontier_size-ThreadLoad(next_frontier_size)))
                {

                    // my_current_frontier_d[original_thid]=thGroupId;
                    // my_current_frontier_d[warpId]=thGroupId;

                    dump[0]=1;
                    dump_local=1;

                    // break;//Don't break here 
                }
            }
            dump_local= __shfl_sync(0xffffffff, dump_local, 0);  
            // dump_local = __any_sync(0xffffffff,dump_local);
            // dump_local = __ballot_sync(0xffffffff, dump_local==1);
            if (dump_local!=0) 
            {
                // printf("Aux: Breaking at prediction logic!\n");               
                my_current_frontier_d[original_thid]=thGroupId;
                #ifdef enable_warp_debug
                printf("Aux_Predict_dumping: original thread:%d my_current_frontier_d[original_thid]: %d\n",original_thid, my_current_frontier_d[original_thid]);
                #endif
                break;
            }
            //Use ballot sync to break.
            // #endif
#ifdef enable_debug
#ifdef all_frontier_checked
            frontierchecked[thGroupId]=1;
#endif
#endif
            for (int k=col_st[front]+laneID; k < col_ed[front]; k +=32)
            {

                int_t m = csr[k];

                if (m > src) 
                {
                    //condition added to avoid the earlier detection of terminating vertices
                    //and to avoid maintain cost for these terminating vertices
                    if (atomicMax(&fill_in_d[fill_in_offset+m],src) < src)
                    {
                        //use cost_array_offset only for fill_in flag. 
                        //The fill-in flag will be disabled later.
                        //If we don't use fill-in flag, we would be recounting the #fill-ins
                        atomicAdd(fill_count,1);
#ifdef enable_debug
                        // printf("source:%d Detected_fill_in:%d\n",src,m);
#endif
                    }
                    // printf("Continuing for larger neighbors!\n");
                    continue;
                }

#ifdef enable_debug
                // printf("col_st[%d]:%d  col_ed[%d]:%d\n ",front,col_st[front],front, col_ed[front]);
#endif
#ifdef enable_debug
                // printf("col_st[%d]:%d  col_ed[%d]:%d\n ",front,col_st[front],front, col_ed[front]);
#endif

                if (cost_array_d[cost_array_offset+m] > cost)
                {  
                    uint_t old_cost=atomicMin(&cost_array_d[cost_array_offset+m],cost);
                    // if (atomicMin(&cost_array_d[cost_array_offset+m],cost) > cost)
                    if (old_cost > cost)
                    {
                        if (m < src)
                        {
                            int_t front_position=atomicAdd(next_frontier_size,1);
#ifdef late_FQ_size_check
                            if (front_position > real_allocation-1)
                            {
                                // my_current_frontier_d[original_thid]=thGroupId;
                                // my_current_frontier_d[warpId]=thGroupId;
                                atomicAdd(next_frontier_size,-1);
                                atomicMax(&cost_array_d[cost_array_offset+m],old_cost);// Making the neighbor avalable for visit in the next
                                //traversal after dump/load opertation
                                dump[0]=1;
                                dump_local=1;
                                // printf("Aux: Breaking at late frontier check!\n");
                                break;
                            }
#endif
                            next_frontier[front_position]=m;
#ifdef enable_debug
                            // printf("Neighbor:%d enqueued with source:%d\n",m,src);
#endif
                            next_src_frontier_d[front_position]=s_id;//src_frontier_d[thid];
                        }

                        if ((m + max_id_offset) > cost)
                        {
                            if (atomicMax(&fill_in_d[fill_in_offset+m],src) < src)
                            {
                                atomicAdd(fill_count,1);
#ifdef enable_debug
                                // printf("source:%d Detected_fill_in:%d\n",src,m);
#endif
                            }
                            // else
                            // {
                            //     continue; //Don't put the fill-in into the frontier queue because its neighbors already have <= the max_id
                            //     // that this fill-in proposes
                            // }

                        }
                    }

                }



            }
#ifdef late_FQ_size_check
            dump_local = __any_sync(0xffffffff,dump_local);//Added for warp-centric
            if (dump_local!=0) 
            {
                my_current_frontier_d[original_thid]=thGroupId;
                #ifdef enable_warp_debug
                printf("AUX_LATE_dumping: original thread:%d my_current_frontier_d[original_thid]: %d\n",original_thid, my_current_frontier_d[original_thid]);
                #endif
                break; // added for the belated FQ size check
            }
#endif
        }
        if (laneID==0) thGroupId = atomicAdd(next_front_d,1);             
        thGroupId= __shfl_sync(0xffffffff, thGroupId, 0);  
        // thid=atomicAdd(next_front_d,1);

    }
    if (dump_local==0) my_current_frontier_d[original_thid]=INT_MAX;
    // if (laneID==0)
    // {
    //     if (dump_local==0) my_current_frontier_d[warpId]=INT_MAX;
    // }

}


__global__ void  Compute_fillins_Auxiliary_level_traversal_thread_centric  (int_t* frontier_size, uint_t* cost_array_d,
    uint_t* fill_in_d,uint_t* frontier,
        uint_t* next_frontier,int_t vert_count,
        int_t* csr, int_t* col_st,  int_t* col_ed,
        ull_t* fill_count,
        uint_t* src_frontier_d,uint_t* next_src_frontier_d,
        int_t* source_d,    int_t* next_frontier_size,
        int_t* lock_d, int_t* dump,int_t* load,
        uint_t max_id_offset, int_t* next_front_d,ull_t passed_allocated_frontier_size,
        int_t* my_current_frontier_d, int_t* current_buffer_m,int_t* next_buffer_m, int_t* offset_kernel,
        int_t* swap_CPU_buffers,int_t* frontierchecked,int_t* swap_GPU_buffers_m,int_t* remaining_frontier_size,
        int_t max_vertex_group,int_t first_source,int_t N_gpu, ull_t allocated_size_max_id,ull_t real_allocation)

{

    //A subset of the original kernel that only finishes the traversal for current group
    //The function is called repeatedly in case of reloding the excess frontiers and 
    //or on writing the excess frontiers

    int_t thid= blockDim.x*blockIdx.x+threadIdx.x;

    int_t original_thid=thid;    
#ifdef enable_debug    
    if (original_thid==0) 
    {
        printf("Inside auxiliary kernel\n");
        // printf("At kernel launch. Aux: next_front_d:%d\n",next_front_d[0]);
    }
#endif

    int_t dump_local=0;
    // int_t first_source=begin_group+gpu_id;//optimization for removing the intermediate bubbles in the cost array
    for(thid=my_current_frontier_d[original_thid] ;thid < frontier_size[0];/*thid+=(gridDim.x*blockDim.x)*/)
        // for(thid=original_thid+offset_kernel[0]; thid < frontier_size[0];/*thid+=(gridDim.x*blockDim.x)*/)
    {

        int_t front=frontier[thid];   
        // printf("Inside auxiliary kernel: original_thid:%d thid:%d front:%d  frontier_size:%d next_frontier_size:%d\n",original_thid, thid,front,frontier_size[0],ThreadLoad(next_frontier_size));    
        // int_t cost_array_offset=src_frontier_d[thid] * vert_count;
        int_t s_id=src_frontier_d[thid];
        int_t cost_array_offset=(s_id*(2*(first_source+1)+(s_id-1)*N_gpu))/2;//optimization for removing the intermediate bubbles in the cost array        
        // int_t cost_array_offset=src_frontier_d[thid] * max_vertex_group;
        int_t fill_in_offset=s_id *vert_count;   
        //  int_t cost = Maximum(front,cost_array_d[cost_array_offset+front]);
        uint_t cost = Maximum(front+max_id_offset,cost_array_d[cost_array_offset+front]);
        int_t src=source_d[s_id];
        // printf(" auxiliary kernel: original_thid: %d  thid:%d  front:%d src:%d frontier_size: %d  next_frontier_size:%d\n",original_thid,thid,front,src,frontier_size[0],ThreadLoad(next_frontier_size));
        if (src < vert_count)
        {          
            // #ifdef early_exit 
            int degree=  col_ed[front]- col_st[front];

            if (degree > (passed_allocated_frontier_size-ThreadLoad(next_frontier_size)))
            {

                my_current_frontier_d[original_thid]=thid;

                dump[0]=1;
                dump_local=1;

                break;
            }
            // #endif
#ifdef enable_debug
#ifdef all_frontier_checked
            frontierchecked[thid]=1;
#endif
#endif
            for (int k=col_st[front]; k < col_ed[front]; k++)
            {

                int_t m = csr[k];

                if (m > src) 
                {
                    //condition added to avoid the earlier detection of terminating vertices
                    //and to avoid maintain cost for these terminating vertices
                    if (atomicMax(&fill_in_d[fill_in_offset+m],src) < src)
                    {
                        //use cost_array_offset only for fill_in flag. 
                        //The fill-in flag will be disabled later.
                        //If we don't use fill-in flag, we would be recounting the #fill-ins
                        atomicAdd(fill_count,1);
#ifdef enable_debug
                        // printf("source:%d Detected_fill_in:%d\n",src,m);
#endif
                    }
                    continue;
                }

#ifdef enable_debug
                // printf("col_st[%d]:%d  col_ed[%d]:%d\n ",front,col_st[front],front, col_ed[front]);
#endif
#ifdef enable_debug
                // printf("col_st[%d]:%d  col_ed[%d]:%d\n ",front,col_st[front],front, col_ed[front]);
#endif

                if (cost_array_d[cost_array_offset+m] > cost)
                {  
                    uint_t old_cost=atomicMin(&cost_array_d[cost_array_offset+m],cost);
                    // if (atomicMin(&cost_array_d[cost_array_offset+m],cost) > cost)
                    if (old_cost > cost)
                    {
                        if (m < src)
                        {
                            int_t front_position=atomicAdd(next_frontier_size,1);
#ifdef late_FQ_size_check
                            if (front_position > real_allocation-1)
                            {
                                my_current_frontier_d[original_thid]=thid;
                                atomicAdd(next_frontier_size,-1);
                                atomicMax(&cost_array_d[cost_array_offset+m],old_cost);// Making the neighbor avalable for visit in the next
                                //traversal after dump/load opertation
                                dump[0]=1;
                                dump_local=1;
                                break;
                            }
#endif
                            next_frontier[front_position]=m;
#ifdef enable_debug
                            // printf("Neighbor:%d enqueued with source:%d\n",m,src);
#endif
                            next_src_frontier_d[front_position]=s_id;//src_frontier_d[thid];
                        }

                        if ((m + max_id_offset) > cost)
                        {
                            if (atomicMax(&fill_in_d[fill_in_offset+m],src) < src)
                            {
                                atomicAdd(fill_count,1);
#ifdef enable_debug
                                // printf("source:%d Detected_fill_in:%d\n",src,m);
#endif
                            }
                            // else
                            // {
                            //     continue; //Don't put the fill-in into the frontier queue because its neighbors already have <= the max_id
                            //     // that this fill-in proposes
                            // }

                        }
                    }

                }



            }
#ifdef late_FQ_size_check
            if (dump_local==1) break; // added for the belated FQ size check
#endif
        }
        thid=atomicAdd(next_front_d,1);

    }
    if (dump_local==0) my_current_frontier_d[original_thid]=INT_MAX;

}
__global__ void  Initialize_current_frontier (int_t* my_current_frontier_d)
{
    int_t thid=blockDim.x*blockIdx.x+threadIdx.x;
    #ifdef thread_centric
    my_current_frontier_d[thid]=thid;// For thread-centric
    #else
    int_t warp_id = thid >> 5;// For warp-centric
    my_current_frontier_d[thid]=warp_id;// For warp-centric
    #endif
}


__global__ void Assign_cost_initializeFQ (uint_t* cost_array_d,
    uint_t* fill_in_d,uint_t* frontier,
        uint_t* next_frontier,int_t vert_count,
        int_t* csr, int_t* col_st,  int_t* col_ed,
        ull_t* fill_count,int_t gpu_id,int_t N_gpu,
        uint_t* src_frontier_d,uint_t* next_src_frontier_d,
        int_t* source_d,  int_t* frontier_size,  int_t* next_frontier_size,
        int_t* lock_d, int_t N_groups, int_t* dump,int_t* load,
        int_t N_src_group, uint_t max_id_offset, 
        int_t* next_front_d, ull_t passed_allocated_frontier_size,
        int_t* my_current_frontier_d,int_t* frontierchecked
        /*, int_t* offset_next_kernel*/,int_t* swap_GPU_buffers_m,
        int_t* count_thread_exiting, int_t begin_group,int_t max_vertex_group,ull_t max_id_allocated_size,int_t first_source,
        ull_t real_allocation,uint_t* Memory_allocated_d)
{

    // int_t first_source=begin_group+gpu_id;//optimization for removing the intermediate bubbles in the cost array
    // int_t thid= blockDim.x*blockIdx.x+threadIdx.x;
    for (uint_t thid=blockDim.x*blockIdx.x+threadIdx.x; thid < N_src_group;thid+=(blockDim.x*gridDim.x))
    {
        // int_t cost_array_offset=thid*max_vertex_group;//int_t cost_array_offset=source_id*vert_count;
        uint_t first_source=begin_group+gpu_id;//optimization for removing the intermediate bubbles in the cost array
        uint_t cost_array_offset=(thid*(2*(first_source+1)+(thid-1)*N_gpu))/2;//optimization for removing the intermediate bubbles in the cost array
        //cost_array_d[cost_array_offset+source_d[thid]]=MIN_VAL;
        // printf("source: %d\n",source_d[thid]);
        // #ifdef debug_illegal_memory_higher_nodes
        // if (source_d[thid] < vert_count)
        // #endif
        if (source_d[thid] < vert_count)
        {
            if ((cost_array_offset+source_d[thid]) > max_id_allocated_size)
            {
                printf("Err! Illegal memory acess by max_id array\n");
            }
            cost_array_d[cost_array_offset+source_d[thid]] = max_id_offset; //max_id_offset=MIN_VAL for the current group
            // printf("Assigning cost to the source:%u\n",source_d[thid]);
        }
    }


    for (int_t src_id=blockIdx.x ; src_id < N_src_group; src_id +=gridDim.x)
    {
        uint_t source=source_d[src_id];
        // if (threadIdx.x == 0)  
        // {
        //     printf("Checking neighbors of the source:%u\n",source);
        // }
        if (source < vert_count)
        {
            // if (threadIdx.x == 0) printf("valid source: %d #neighbors:%d\n",source,col_ed[source]-col_st[source]);
            for (int_t b_tid=threadIdx.x+col_st[source]; b_tid < col_ed[source]; b_tid+=blockDim.x)
            {
                uint_t neighbor=csr[b_tid];
                // printf("neighbor:%u\n",neighbor);
                // printf("Neighbor: %d\n",neighbor);
                // int_t cost_array_offset=src_id*max_vertex_group;//int_t cost_array_offset=source_id*vert_count;
                uint_t cost_array_offset=(src_id*(2*(first_source+1)+(src_id-1)*N_gpu))/2;
                uint_t fill_in_offset=src_id*vert_count;
                // cost_array_d[cost_array_offset+ neighbor]=MIN_VAL;//Initialising all the neighbots costs with minimum cost

                fill_in_d[fill_in_offset+ neighbor]=source;
                if (neighbor < source) 
                {
                    if ((cost_array_offset+neighbor) > max_id_allocated_size)
                    {
                        printf("Err! Illegal memory acess by max_id array\n");
                    }
                    cost_array_d[cost_array_offset+ neighbor]=max_id_offset;
                    // printf("Putting neighbor into the frontier queue!\n");
                    uint_t front_position=atomicAdd(frontier_size,1);
                    // printf("Detected frontier! new frontier size: %d\n", front_position+1);
                    // time_atomic+=(clock64()-start_time_atomic);
                    frontier[front_position]=neighbor;
                    src_frontier_d[front_position]=src_id;//save the source position in the array not the source itself
                    // if (source==1020)  printf("NOTE:groupwise src:%d has neighbor:%d \n",source,neighbor);
                }           
            }
        }
    }
#ifdef enable_debug
    if (blockDim.x*blockIdx.x+threadIdx.x==0) printf("Initial frontier size: %d\n",frontier_size[0]);
#endif
}

__global__ void Compute_fillins_joint_traversal_group_wise_level_traversal_thread_centric (uint_t* cost_array_d,
    uint_t* fill_in_d,uint_t* frontier,
        uint_t* next_frontier,int_t vert_count,
        int_t* csr, int_t* col_st,  int_t* col_ed,
        ull_t* fill_count,int_t gpu_id,int_t N_gpu,
        uint_t* src_frontier_d,uint_t* next_src_frontier_d,
        int_t* source_d,  int_t* frontier_size,  int_t* next_frontier_size,
        int_t* lock_d, int_t N_groups, int_t* dump,int_t* load,
        int_t N_src_group,uint_t max_id_offset, 
        int_t* next_front_d, ull_t passed_allocated_frontier_size,
        int_t* my_current_frontier_d, int_t* frontierchecked
        /*, int_t* offset_next_kernel*/, int_t* swap_GPU_buffers_m,
        int_t* count_thread_exiting, int_t* remaining_frontier_size,
        int_t begin_group,int_t max_vertex_group,ull_t max_id_allocated_size,
        int_t first_source,ull_t real_allocation,uint_t* Memory_allocated_d)
{

    // printf("Inside groupwise kernel\n");
    int_t level=0; //test for average frontier size
    uint_t thid= blockDim.x*blockIdx.x+threadIdx.x;
    uint_t original_thid=thid;    
    int_t dump_local=0;
    // volatile int *remaining_frontier_size_volatile = remaining_frontier_size;
    my_current_frontier_d[original_thid]=INT_MAX;
    // if (group==19) 
    // {
    //     int a=5;
    //     printf("a:%d\n",a);
    // }
    // int_t first_source=begin_group+gpu_id;//optimization for removing the intermediate bubbles in the cost array
    for(thid=original_thid;thid < frontier_size[0];/*thid+=(gridDim.x*blockDim.x)*/)
    {
        uint_t front=frontier[thid];
        uint_t s_id=src_frontier_d[thid];
        uint_t cost_array_offset=(s_id*(2*(first_source+1)+(s_id-1)*N_gpu))/2;//optimization for removing the intermediate bubbles in the cost array
        // int_t cost_array_offset=src_frontier_d[thid] * max_vertex_group;
        int_t fill_in_offset=s_id * vert_count;
        //  int_t cost = Maximum(front,cost_array_d[cost_array_offset+front]);
        uint_t cost = Maximum(front+max_id_offset,cost_array_d[cost_array_offset+front]);
        uint_t src = source_d[s_id];

        // int_t prune=0;
        if (src < vert_count)
        {  
            //                #ifdef early_exit  
            int degree =  col_ed[front]- col_st[front];
            // int_t remaining_space = atomicAdd(remaining_frontier_size,-degree);  
            // if (remaining_space < 0)
            // if ((last_space-degree) < 0)
            // if (degree > (ThreadLoad(remaining_frontier_size)))
            if (degree > (passed_allocated_frontier_size-ThreadLoad(next_frontier_size)))
            {
                //exit taking consideration of worst case
                my_current_frontier_d[original_thid]=thid;
#ifdef enable_debug
                // if (original_thid==0) 
                // printf("Earlier exit groupwise original_thid:%d  thid while dumping:%d  front:%d  src:%d\n",original_thid,thid,front,src);
                // printf("Earlier exit groupwise original_thid:%d  \n",original_thid);
#endif
                // printf("Earlier exit groupwise original_thid:%d  \n",original_thid);
                // sync_to_CPU(next_frontier,next_frontier_size[0]);
                //  if (dump[0]!=1) dump[0]=1;
                // dump[0]=1;
                // dump_volatile[0]=1;
                dump[0]=1;
                dump_local=1;
                // atomicAdd(remaining_frontier_size,degree);
                // cost_array_d[cost_array_offset+m]=old_cost;//Reassigning the old cost
                // atomicMax(offset_next_kernel,thid);
                // return;
                break;
            }

            // #endif    
            // printf("Continuing for fill-in/frontier check\n");
#ifdef enable_debug
#ifdef all_frontier_checked
            frontierchecked[thid]=1;
#endif  
#endif
            for (int k=col_st[front]; k < col_ed[front]; k++)
            {
                uint_t m = csr[k];

                if (m > src) 
                {
                    //condition added to avoid the earlier detection of terminating vertices
                    //and to avoid maintain cost for these terminating vertices
                    if (atomicMax(&fill_in_d[fill_in_offset+m],src) < src)
                    {
                        //use cost_array_offset only for fill_in flag. 
                        //The fill-in flag will be disabled later.
                        //If we don't use fill-in flag, we would be recounting the #fill-ins
                        atomicAdd(fill_count,1);
                        // printf("source:%d Detected_fill_in:%d\n",src,m);
#ifdef enable_debug
                        // printf("source:%d Detected_fill_in:%d\n",src,m);
#endif
                    }
                    continue;
                }

                // if ((src==1020) && (m==994)) printf("NOTE:groupwise_no_costupdate src:%d neighbor:%d front:%d proposed_cost:%u original_cost:%u\n",src,m,front,cost,cost_array_d[cost_array_offset+m]);
                //  if ((src==1020) && (front==997)) printf("NOTE:groupwise src:%d neighbor:%d front:%d\n",src,m,front);
                // if ((cost_array_offset + m) > max_id_allocated_size)
                // {
                //     printf("Err! Illegal memory acess by max_id array\n");
                // }
                if (cost_array_d[cost_array_offset+m] > cost)
                {
                    uint_t old_cost=atomicMin(&cost_array_d[cost_array_offset+m],cost);
                    // if (atomicMin(&cost_array_d[cost_array_offset+m],cost) > cost)
                    if (old_cost > cost)
                    {

                        if (m < src)
                        {

                            int_t front_position=atomicAdd(next_frontier_size,1);
#ifdef late_FQ_size_check
                            if (front_position > real_allocation-1)
                            {
                                my_current_frontier_d[original_thid]=thid;
                                atomicAdd(next_frontier_size,-1);
                                atomicMax(&cost_array_d[cost_array_offset+m],old_cost);// Making the neighbor avalable for visit in the next
                                //traversal after dump/load opertation

                                dump[0]=1;
                                dump_local=1;

                                break;
                            }
#endif
                            // printf("front_position: %d \n",front_position);
                            next_frontier[front_position]=m;
                            next_src_frontier_d[front_position]=s_id;//src_frontier_d[thid];

                        }

                        if ((m + max_id_offset) > cost)
                        {

                            if (atomicMax(&fill_in_d[fill_in_offset+m],src) < src)
                            {
                                atomicAdd(fill_count,1);
#ifdef enable_debug
                                // printf("source:%d Detected_fill_in:%d\n",src,m);
#endif
                            }
                        }
                    }
                }
            }
#ifdef late_FQ_size_check
            if (dump_local==1) break; // added for the belated FQ size check
#endif
        }
        thid=atomicAdd(next_front_d,1);
    }
}
__global__ void Compute_fillins_joint_traversal_group_wise_level_traversal (uint_t* cost_array_d,
    uint_t* fill_in_d,uint_t* frontier,
        uint_t* next_frontier,int_t vert_count,
        int_t* csr, int_t* col_st,  int_t* col_ed,
        ull_t* fill_count,int_t gpu_id,int_t N_gpu,
        uint_t* src_frontier_d,uint_t* next_src_frontier_d,
        int_t* source_d,  int_t* frontier_size,  int_t* next_frontier_size,
        int_t* lock_d, int_t N_groups, int_t* dump,int_t* load,
        int_t N_src_group,uint_t max_id_offset, 
        int_t* next_front_d, ull_t passed_allocated_frontier_size,
        int_t* my_current_frontier_d, int_t* frontierchecked
        /*, int_t* offset_next_kernel*/, int_t* swap_GPU_buffers_m,
        int_t* count_thread_exiting, int_t* remaining_frontier_size,
        int_t begin_group,int_t max_vertex_group,ull_t max_id_allocated_size,
        int_t first_source,ull_t real_allocation,uint_t* Memory_allocated_d)
{

    // printf("Inside groupwise kernel\n");
    int_t level=0; //test for average frontier size
    uint_t thid= blockDim.x*blockIdx.x+threadIdx.x;
    uint_t original_thid=thid;    
    int_t laneID = threadIdx.x & 0x1f;
    int_t warpId = thid >> 5;
    uint_t dump_local=0;
    // volatile int *remaining_frontier_size_volatile = remaining_frontier_size;
    my_current_frontier_d[original_thid]=INT_MAX;
    // my_current_frontier_d[warpId]=INT_MAX;
    // if (group==19) 
    // {
    //     int a=5;
    //     printf("a:%d\n",a);
    // }

    // for(thid=original_thid;thid < frontier_size[0];)
    // if (thid ==0) printf("Inside groupwise kernel\n");
    for(int_t thGroupId = warpId; thGroupId < frontier_size[0];)
    {
        uint_t front=frontier[thGroupId];
        uint_t s_id=src_frontier_d[thGroupId];
        uint_t cost_array_offset=(s_id*(2*(first_source+1)+(s_id-1)*N_gpu))/2;//optimization for removing the intermediate bubbles in the cost array
        // int_t cost_array_offset=src_frontier_d[thid] * max_vertex_group;
        int_t fill_in_offset=s_id * vert_count;
        //  int_t cost = Maximum(front,cost_array_d[cost_array_offset+front]);
        uint_t cost = Maximum(front+max_id_offset,cost_array_d[cost_array_offset+front]);
        uint_t src = source_d[s_id];

        // int_t prune=0;
        if (src < vert_count)
        {  
            //                #ifdef early_exit  
            int degree =  col_ed[front]- col_st[front];
            // int_t remaining_space = atomicAdd(remaining_frontier_size,-degree);  
            // if (remaining_space < 0)
            // if ((last_space-degree) < 0)
            // if (degree > (ThreadLoad(remaining_frontier_size)))
            if (laneID==0)
            {
                
                if (degree > (passed_allocated_frontier_size-ThreadLoad(next_frontier_size)))
                {
                    //exit taking consideration of worst case
                    // my_current_frontier_d[original_thid]=thGroupId;
                    // my_current_frontier_d[warpId]=thGroupId;
#ifdef enable_debug
                    // if (original_thid==0) 
                    // printf("Earlier exit groupwise original_thid:%d  thid while dumping:%d  front:%d  src:%d\n",original_thid,thid,front,src);
                    // printf("Earlier exit groupwise original_thid:%d  \n",original_thid);
                    printf("Earlier exit groupwise original_thid:%d  \n",original_thid);
#endif
                    
                    // sync_to_CPU(next_frontier,next_frontier_size[0]);
                    //  if (dump[0]!=1) dump[0]=1;
                    // dump[0]=1;
                    // dump_volatile[0]=1;
                    dump[0]=1;
                    dump_local=1;
                    // atomicAdd(remaining_frontier_size,degree);
                    // cost_array_d[cost_array_offset+m]=old_cost;//Reassigning the old cost
                    // atomicMax(offset_next_kernel,thid);
                    // return;
                    // break;
                }
            }
            // printf("Before:lane %d  dump_local:%d  \n",laneID, dump_local);
            dump_local= __shfl_sync(0xffffffff, dump_local, 0);  
            // dump_local = __any_sync(0xffffffff,dump_local);
            // dump_local = __ballot_sync(0xffffffff, dump_local==1);
            if (dump_local==1) 
            {
                // printf("After:lane %d  dump_local:%d  \n",laneID, dump_local);
                my_current_frontier_d[original_thid]=thGroupId;
                #ifdef enable_warp_debug
                printf("Group_Predict: original thread:%d my_current_frontier_d[original_thid]:%d \n",original_thid,my_current_frontier_d[original_thid]);
                #endif
                // printf("Breaking at prediction logic!\n");
                break;
            }

            // #endif    
            // printf("Continuing for fill-in/frontier check\n");
#ifdef enable_debug
#ifdef all_frontier_checked
            frontierchecked[thGroupId]=1;
#endif  
#endif
            // for (int k=col_st[front]; k < col_ed[front]; k++)
            for (int k = col_st[front] + laneID; k < col_ed[front]; k += 32)
            {
                uint_t m = csr[k];

                if (m > src) 
                {
                    //condition added to avoid the earlier detection of terminating vertices
                    //and to avoid maintain cost for these terminating vertices
                    if (atomicMax(&fill_in_d[fill_in_offset+m],src) < src)
                    {
                        //use cost_array_offset only for fill_in flag. 
                        //The fill-in flag will be disabled later.
                        //If we don't use fill-in flag, we would be recounting the #fill-ins
                        atomicAdd(fill_count,1);
                        // printf("source:%d Detected_fill_in:%d\n",src,m);
#ifdef enable_debug
                        // printf("source:%d Detected_fill_in:%d\n",src,m);
#endif
                    }
                    // printf("Continuing when large vertex detected\n");
                    continue;
                }

                // if ((src==1020) && (m==994)) printf("NOTE:groupwise_no_costupdate src:%d neighbor:%d front:%d proposed_cost:%u original_cost:%u\n",src,m,front,cost,cost_array_d[cost_array_offset+m]);
                //  if ((src==1020) && (front==997)) printf("NOTE:groupwise src:%d neighbor:%d front:%d\n",src,m,front);
                // if ((cost_array_offset + m) > max_id_allocated_size)
                // {
                //     printf("Err! Illegal memory acess by max_id array\n");
                // }
                if (cost_array_d[cost_array_offset+m] > cost)
                {
                    uint_t old_cost=atomicMin(&cost_array_d[cost_array_offset+m],cost);
                    // if (atomicMin(&cost_array_d[cost_array_offset+m],cost) > cost)
                    if (old_cost > cost)
                    {

                        if (m < src)
                        {

                            int_t front_position=atomicAdd(next_frontier_size,1);
#ifdef late_FQ_size_check
                            if (front_position > real_allocation-1)
                            {
                                // my_current_frontier_d[original_thid]=thGroupId;
                                // my_current_frontier_d[warpId]=thGroupId;
                                atomicAdd(next_frontier_size,-1);
                                atomicMax(&cost_array_d[cost_array_offset+m],old_cost);// Making the neighbor avalable for visit in the next
                                //traversal after dump/load opertation

                                dump[0]=1;
                                dump_local=1;
                                // printf("Group: Breaking late \n");
                                // printf("Group_late: Breaking at late frontier check!\n");
                                #ifdef enable_warp_debug
                                printf("Group_late: original thread:%d my_current_frontier_d[original_thid]:%d \n",original_thid,my_current_frontier_d[original_thid]);
                                #endif
                                break;
                            }
#endif
                            // printf("front_position: %d \n",front_position);
                            next_frontier[front_position]=m;
                            next_src_frontier_d[front_position]=s_id;//src_frontier_d[thid];

                        }

                        if ((m + max_id_offset) > cost)
                        {

                            if (atomicMax(&fill_in_d[fill_in_offset+m],src) < src)
                            {
                                atomicAdd(fill_count,1);
#ifdef enable_debug
                                // printf("source:%d Detected_fill_in:%d\n",src,m);
#endif
                            }
                        }
                    }
                }
            }
#ifdef late_FQ_size_check
            dump_local = __any_sync(0xffffffff,dump_local);//Added for warp-centric
            if (dump_local!=0) 
            {
                my_current_frontier_d[original_thid]=thGroupId;
                break; // added for the belated FQ size check
            }
#endif
        }
        // thid=atomicAdd(next_front_d,1);
        if (laneID==0) thGroupId = atomicAdd(next_front_d,1);

        thGroupId= __shfl_sync(0xffffffff, thGroupId, 0);  
    }
}
 
__global__ void Assign_sources (int_t gpu_id,int_t N_gpu,
        int_t* source_d, 
        int_t N_src_group, int_t begin_group)
{

    int_t thid= blockDim.x*blockIdx.x+threadIdx.x;

    while (thid < N_src_group)
    {
        source_d[thid]=gpu_id + thid * N_gpu + begin_group; //Interleave the sources among the GPUs
        // source_d[thid]=gpu_id + thid * N_gpu + group * N_src_group; //Interleave the sources among the GPUs
        // printf("gpu_id:%d  thid:%d  N_gpu:%d  begin_group:%d N_src_group:%d source_d[thid:%d\n",gpu_id,thid,N_gpu,begin_group,N_src_group,source_d[thid]);
        thid+=(blockDim.x*gridDim.x);
        
    }
}

int Compute_Src_group_original(int vert_count)
{
    // cout<<"Start finding N_src_per_group"<<endl;
    // double temp = 2147483648/(double)(6*vert_count);
    double temp = 3758096384/(double)(5*vert_count);
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
    // cout<<"Finished finding N_src_per_group"<<endl;
    return (int)temp;
}


int Compute_Src_group(int vert_count,float total_memory_allocation_in_GB)
{
    // cout<<"Start finding N_src_per_group"<<endl;
    // double temp = 2147483648/(double)(6*vert_count);
    ull_t N_elemnts = (total_memory_allocation_in_GB*1024*1024*1024)/4;
    cout<<"Computed N_ints/unsigned ints: "<<N_elemnts<<endl;
    double temp =  N_elemnts/(double)(6*vert_count);
    // double temp = 3758096384/(double)(5*vert_count);
    //Converging to the power of 2 value just smaller than temp


    // if (temp > vert_count)
    // {
    //     temp=(int)log_2(vert_count/(double)2);
    //     temp=pow(2,temp);

    // }
    // else
    // {
    //     temp=(int)log_2(temp);
    //     temp=pow(2,temp);
    // }
  
    return (int)temp;
}

void  SWAP_CPU_BUFFERS(uint_t* &CPU_buffer_next_level,uint_t* &CPU_buffer_next_level_source,int_t& size_CPU_buffer_next_level,
        uint_t* &CPU_buffer_current_level,uint_t*  &CPU_buffer_current_level_source,int_t& size_CPU_buffer_current_level,
        int_t* &current_N_buffer_CPU,int_t* &next_N_buffer_CPU,int_t* swap_CPU_buffers_m)
{
    // swap_ptr_index(CPU_buffer_next_level,CPU_buffer_current_level);
    uint_t* temp=CPU_buffer_next_level;
    CPU_buffer_next_level=CPU_buffer_current_level;
    CPU_buffer_current_level=temp;

    // swap_ptr_index(CPU_buffer_next_level_source,CPU_buffer_current_level_source);
    temp=CPU_buffer_next_level_source;
    CPU_buffer_next_level_source=CPU_buffer_current_level_source;
    CPU_buffer_current_level_source=temp;

    int_t* temp_int =next_N_buffer_CPU;
    next_N_buffer_CPU=current_N_buffer_CPU;
    current_N_buffer_CPU=temp_int;

    size_CPU_buffer_current_level=size_CPU_buffer_next_level;
    size_CPU_buffer_next_level=0;
    swap_CPU_buffers_m[0]=0;

}

void SWAP_GPU_BUFFERS(uint_t* &frontier_d,uint_t* &next_frontier_d,uint_t* &src_frontier_d,uint_t* &next_src_frontier_d,int_t* &swap_GPU_buffers_m)
{
    uint_t* temp=frontier_d;
    frontier_d=next_frontier_d;
    next_frontier_d=temp;

    temp=src_frontier_d;
    src_frontier_d=next_src_frontier_d;
    next_src_frontier_d=temp;
    swap_GPU_buffers_m[0]=0;

}

void Display_CPU_buffers(int_t size, int_t* CPU_buffer_frontier, int_t* CPU_buffer_source,int_t flag_load_dump,ull_t temp_allocated_frontier_size)
{
    std::string activity;
    if (flag_load_dump==0)
    {
        activity=" dumping  ";
    }
    else
    {
        activity=" loading  ";
    }
    int_t lot=0;
    for (int i=0;i<size;i++)
    {
        if (i%temp_allocated_frontier_size==0) cout<<endl<<"lot:"<<lot++<<activity;
        cout<<i<<":"<<CPU_buffer_frontier[i]<<"/"<<CPU_buffer_source[i]<<"  ";

    }
    cout<<endl;
}

void Compute_fillins_joint_traversal_group_wise_splitted (uint_t* cost_array_d,
    uint_t* fill_in_d,uint_t* frontier_d,
        uint_t* next_frontier_d,int_t vert_count,
        int_t* csr_d, int_t* col_st_d,  int_t* col_ed_d,
        ull_t* fill_count_d,int_t myrank,int_t num_process,
        uint_t* src_frontier_d,uint_t* next_src_frontier_d,
        int_t* source_d,  int_t* frontier_size_d,  int_t* next_frontier_size_d,
        int_t* lock_d, int_t N_groups, int_t* dump_m,int_t* load_m,
        int_t N_src_group/*,int_t group*/,uint_t max_id_offset, int_t* next_front_d,ull_t temp_allocated_frontier_size,
        int_t* my_current_frontier_d,int_t* frontierchecked
        /*, int_t* offset_last_kernel*/,int_t* swap_GPU_buffers_m,
        int_t*   count_thread_exiting, int_t BLKS_NUM,int_t blockSize,int_t* remaining_frontier_size,
        int_t begin_group,int_t max_vertex_group,ull_t allocated_size_max_id, int_t first_source,ull_t real_allocation,
        uint_t* Memory_allocated_d,int_t Nwarp)
{
    int_t frontier_size_h;
    int_t next_front_default_offset;
    #ifdef thread_centric
    next_front_default_offset=BLKS_NUM*blockSize;
    #else
    next_front_default_offset=Nwarp;
    #endif
    Assign_sources <<<BLKS_NUM,blockSize>>> ( myrank, num_process, source_d, 
            N_src_group,  begin_group);
    // H_ERR(cudaDeviceSynchronize());
    // cout<<"Start assigning the cost to neighbors and fill the intial FQ"<<endl;
    Assign_cost_initializeFQ <<<BLKS_NUM,blockSize>>>  (cost_array_d,fill_in_d,frontier_d,
            next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,myrank,num_process,
            src_frontier_d,next_src_frontier_d,source_d, frontier_size_d, next_frontier_size_d,
            lock_d,N_groups,
            dump_m,load_m,
            N_src_group/*,group*/,max_id_offset,next_front_d,
            temp_allocated_frontier_size,my_current_frontier_d,frontierchecked,
            swap_GPU_buffers_m,count_thread_exiting,
            begin_group, max_vertex_group, allocated_size_max_id,first_source,real_allocation,
            Memory_allocated_d); 
    H_ERR(cudaDeviceSynchronize());//It is necessary
    // cout<<"Finished assigning the cost to neighbors and fill the intial FQ"<<endl;
    H_ERR(cudaMemcpy(&frontier_size_h,frontier_size_d,sizeof(int_t),cudaMemcpyDeviceToHost));
    // cout<<"Frontier size detected: "<<frontier_size_h<<endl;
    if (maximum_front_size < frontier_size_h) maximum_front_size=frontier_size_h;//checking maximum frontier size
    while (frontier_size_h!=0)
    {
        // cout<<"Entering Compute_fillins_joint_traversal_group_wise_level_traversal !"<<endl;
        #ifdef thread_centric
        Compute_fillins_joint_traversal_group_wise_level_traversal_thread_centric<<<BLKS_NUM,blockSize>>>  (cost_array_d,fill_in_d,frontier_d,
            next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,myrank,num_process,
            src_frontier_d,next_src_frontier_d,source_d, frontier_size_d, next_frontier_size_d,
            lock_d,N_groups,
            dump_m,load_m,
            N_src_group/*,group*/,max_id_offset,next_front_d,
            temp_allocated_frontier_size,my_current_frontier_d,frontierchecked,
            swap_GPU_buffers_m,count_thread_exiting, remaining_frontier_size,
            begin_group, max_vertex_group, allocated_size_max_id,first_source, real_allocation,
            Memory_allocated_d);
        #else
        Compute_fillins_joint_traversal_group_wise_level_traversal<<<BLKS_NUM,blockSize>>>  (cost_array_d,fill_in_d,frontier_d,
                next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,myrank,num_process,
                src_frontier_d,next_src_frontier_d,source_d, frontier_size_d, next_frontier_size_d,
                lock_d,N_groups,
                dump_m,load_m,
                N_src_group/*,group*/,max_id_offset,next_front_d,
                temp_allocated_frontier_size,my_current_frontier_d,frontierchecked,
                swap_GPU_buffers_m,count_thread_exiting, remaining_frontier_size,
                begin_group, max_vertex_group, allocated_size_max_id,first_source, real_allocation,
                Memory_allocated_d);
                #endif
        H_ERR(cudaDeviceSynchronize());//It is necessary
        // cout<<"Exited Compute_fillins_joint_traversal_group_wise_level_traversal !"<<endl;
        if (dump_m[0] ==1)
            // if (dump[0]==1)
        {
#ifdef enable_debug
            printf("Dumping the frontier!\n");
#endif
            break;
            // break;
        }
        //not a case of dumping swap the frontiers and contunue to the next loop
        swap_ptr_index(frontier_d,next_frontier_d);
        swap_ptr_index(src_frontier_d,next_src_frontier_d);



        swap_GPU_buffers_m[0] ^=1;
        H_ERR(cudaMemcpy(frontier_size_d,next_frontier_size_d,sizeof(int_t),cudaMemcpyDeviceToDevice));
        H_ERR(cudaMemset(next_frontier_size_d, 0, sizeof(int_t)));
        H_ERR(cudaMemcpy(next_front_d,&next_front_default_offset,sizeof(int_t),cudaMemcpyHostToDevice));

        H_ERR(cudaMemcpy(&frontier_size_h,frontier_size_d,sizeof(int_t),cudaMemcpyDeviceToHost));
        // H_ERR(cudaMemcpy(remaining_frontier_size,&temp_allocated_frontier_size,sizeof(int_t),cudaMemcpyHostToDevice));

        // #ifdef enable_debug
        // printf("frontier size in the last loop: %d allocated frontier size:%d\n dump:%d",frontier_size[0],passed_allocated_frontier_size,dump[0]);
        // printf("Number of fill-ins detected till now: %d\n",fill_count[0]);
        //  #endif
        // frontier_size[0]=next_frontier_size[0];
        // next_frontier_size[0]=0;
        // next_front_d[0] = blockDim.x*gridDim.x;

        H_ERR(cudaDeviceSynchronize());//It is necessary

        //SWAPS Copy next frontiers to current:
        // if (frontier_size_h > real_allocation) printf("Overflow of frontiers from allocated size:%d\n",real_allocation);

    }

    // Compute_fillins_joint_traversal_group_wise <<<BLKS_NUM,blockSize>>>  (cost_array_d,fill_in_d,frontier_d,
    //     next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,myrank,num_process,
    //     src_frontier_d,next_src_frontier_d,source_d, frontier_size_d, next_frontier_size_d,
    //     lock_d,N_groups,
    //     dump_m,load_m,
    //     N_src_group,group,max_id_offset,next_front_d,
    //     temp_allocated_frontier_size,my_current_frontier_d,frontierchecked,
    //     swap_GPU_buffers_m,count_thread_exiting); 
    //                                        H_ERR(cudaDeviceSynchronize());
}

void Compute_fillins_Auxiliary_splitted (int_t* frontier_size_d, uint_t* cost_array_d, uint_t* fill_in_d, uint_t* frontier_d,
        uint_t* next_frontier_d, int_t vert_count, int_t* csr_d, int_t* col_st_d, int_t* col_ed_d,ull_t* fill_count_d,
        uint_t* src_frontier_d, uint_t*  next_src_frontier_d,int_t*  source_d,  int_t* next_frontier_size_d,
        int_t* lock_d,
        int_t* dump_m,int_t* load_m,                           
        uint_t max_id_offset,    
        int_t* next_front_d,ull_t temp_allocated_frontier_size,int_t* my_current_frontier_d, int_t* current_buffer_m, int_t* next_buffer_m,
        int_t* offset_kernel,
        /*int_t group,*/int_t* swap_CPU_buffers_m,int_t* frontierchecked,int_t* swap_GPU_buffers_m,int_t BLKS_NUM,int_t blockSize,
        int_t* remaining_frontier_size,int_t max_vertex_group,int_t first_source,int_t num_process,ull_t allocated_size_max_id,ull_t real_allocation,
        int_t Nwarp)
{
    // int_t next_front_default_offset=BLKS_NUM*blockSize;
    int_t next_front_default_offset;
    #ifdef thread_centric
     next_front_default_offset=BLKS_NUM*blockSize;
     #else
     next_front_default_offset=Nwarp;
    #endif
    int_t frontier_size_h;
    H_ERR(cudaMemcpy(next_front_d,offset_kernel,sizeof(int_t),cudaMemcpyDeviceToDevice));
    // printf("Inside auxiliary kernel: level loop\n");

    H_ERR(cudaMemcpy(&frontier_size_h,frontier_size_d,sizeof(int_t),cudaMemcpyDeviceToHost));
    // printf("frontier_size: %d \n",frontier_size_h);

    while (frontier_size_h!=0)
    {
        // cout<<"Entering Compute_fillins_Auxiliary_level_traversal !"<<endl;
        #ifdef thread_centric
        Compute_fillins_Auxiliary_level_traversal_thread_centric <<<BLKS_NUM,blockSize>>> (frontier_size_d, cost_array_d, fill_in_d, frontier_d,
            next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,
            src_frontier_d,next_src_frontier_d,source_d,  next_frontier_size_d,
            lock_d,
            dump_m,load_m,                           
            max_id_offset,    
            next_front_d,temp_allocated_frontier_size,my_current_frontier_d,current_buffer_m,next_buffer_m,offset_kernel,
            /*group,*/swap_CPU_buffers_m,frontierchecked,swap_GPU_buffers_m,remaining_frontier_size,max_vertex_group, first_source,num_process,
            allocated_size_max_id, real_allocation); 
        #else
        Compute_fillins_Auxiliary_level_traversal <<<BLKS_NUM,blockSize>>> (frontier_size_d, cost_array_d, fill_in_d, frontier_d,
                next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,
                src_frontier_d,next_src_frontier_d,source_d,  next_frontier_size_d,
                lock_d,
                dump_m,load_m,                           
                max_id_offset,    
                next_front_d,temp_allocated_frontier_size,my_current_frontier_d,current_buffer_m,next_buffer_m,offset_kernel,
                /*group,*/swap_CPU_buffers_m,frontierchecked,swap_GPU_buffers_m,remaining_frontier_size,max_vertex_group, first_source,num_process,
                allocated_size_max_id, real_allocation); 
                #endif
        H_ERR(cudaDeviceSynchronize()); 
        // cout<<"Exited Compute_fillins_Auxiliary_level_traversal !"<<endl;
        // printf("Outside auxiliary kernel\n");
        if (dump_m[0] ==1)           
        {
            // if (dump_local==0) my_current_frontier_d[original_thid]=INT_MAX;
            break;     
        }

        Initialize_current_frontier<<<BLKS_NUM,blockSize>>> (my_current_frontier_d);
        //Don't need to synchronize the device with the host here
        if (current_buffer_m[0] > 0)
        {
            //load from the CPU memory

            // my_current_frontier_d[original_thid]=original_thid;
            load_m[0]=1;
            // offset_next_kernel[0]= (blockDim.x*gridDim.x) 
            break;
        }
        else
        {
            //SWAP and make current buffer as the next buffer
            swap_ptr_index(frontier_d,next_frontier_d);
            swap_ptr_index(src_frontier_d,next_src_frontier_d);
            // swap_ptr_index_uint_16(src_frontier_d,next_src_frontier_d);


            swap_GPU_buffers_m[0] ^=1;
            H_ERR(cudaMemcpy(frontier_size_d,next_frontier_size_d,sizeof(int_t),cudaMemcpyDeviceToDevice));
            H_ERR(cudaMemset(next_frontier_size_d, 0, sizeof(int_t)));
            H_ERR(cudaMemcpy(next_front_d,&next_front_default_offset,sizeof(int_t),cudaMemcpyHostToDevice));

            H_ERR(cudaMemcpy(&frontier_size_h,frontier_size_d,sizeof(int_t),cudaMemcpyDeviceToHost));
            H_ERR(cudaMemset(offset_kernel, 0, sizeof(int_t)));
            // H_ERR(cudaMemcpy(remaining_frontier_size,&temp_allocated_frontier_size,sizeof(int_t),cudaMemcpyHostToDevice));
            current_buffer_m[0]=next_buffer_m[0];
            next_buffer_m[0]=0;
            swap_CPU_buffers_m[0]=1;    
            // my_current_frontier_d[original_thid]=original_thid;

            if ((frontier_size_h==0) && (current_buffer_m[0]>0))
            {                    
                load_m[0]=1; 
                break;
            }

        }
        // H_ERR(cudaMemcpy(&frontier_size_h,frontier_size_d,sizeof(int_t),cudaMemcpyDeviceToHost));
    }


}  

int_t Get_Max_id_efficent_N_SourceGroup_original(int_t N_src_group,int_t end_group, int_t N_gpu, int_t begin_group,
        ull_t max_id_size_threshhold, ull_t& allocated_size_max_id)
{
    // int_t groupsizefixed=0;
    //Formula
    // sum = n(a1+an)/2 where a1 = first term, an=last term and n is number of elements
    //~Formula
    unsigned int first_term= (begin_group+1+N_gpu) + (end_group+1);
    //here a1 =begin_group+1+N_gpu the #max_ids for the first source of the last GPU.
    //and an=end_group+1 the #max_ids for the last source of the last GPU
    while (1)
    {
        allocated_size_max_id=(first_term * (ull_t)N_src_group)/2;
        ull_t before_space= (unsigned)N_src_group * (unsigned)(end_group);
        // cout<<"Before space demanded "<< before_space<<endl;
        // if ((unsigned)N_src_group * (unsigned)(end_group) <= allocated_size)

        // max_id_size_threshhold

        // if (( first_term * (ull_t)N_src_group)/2 <= allocated_size)
        if ( allocated_size_max_id <= max_id_size_threshhold)
        {

            // cout<<"After space demanded: "<< allocated_size_max_id <<"  " << "available size: "<< max_id_size_threshhold << endl;
            break;
        }
        else
        {
            // cout<<"After space demanded: "<< allocated_size_max_id <<"  "<< "available size: "<< max_id_size_threshhold << endl;
            N_src_group /= 2;
            // cout<<"New reduced N_src_group: "<<N_src_group<<endl;
        }
    }
    // allocated_size_max_id=space_demanded;
    return N_src_group;

}
int_t Get_Max_id_efficent_N_SourceGroup(int_t N_src_group,int_t end_group, int_t N_gpu, int_t begin_group,
    ull_t maxId_fill_size_threshhold, ull_t& allocated_size_max_id,int_t vert_count)
{
// int_t groupsizefixed=0;
//Formula
// sum = n(a1+an)/2 where a1 = first term, an=last term and n is number of elements
//~Formula
// unsigned int first_term= (begin_group+1+N_gpu) + (end_group+1);

//here a1 =begin_group+1+N_gpu the #max_ids for the first source of the last GPU.
//and an=end_group+1 the #max_ids for the last source of the last GPU
unsigned int first_term;
while (1)
{
    first_term= (begin_group+1+N_gpu) + (end_group+1);
    ull_t allocated_size_maxId_fill=(first_term * (ull_t)N_src_group)/2 + (ull_t)N_src_group *vert_count; //space of both max + fill is allowed to push upto 90% of total memory
    // ull_t before_space= (unsigned)N_src_group * (unsigned)(end_group);
    // cout<<"Before space demanded "<< before_space<<endl;
    // if ((unsigned)N_src_group * (unsigned)(end_group) <= allocated_size)

    // max_id_size_threshhold

    // if (( first_term * (ull_t)N_src_group)/2 <= allocated_size)
    if ( allocated_size_maxId_fill <= maxId_fill_size_threshhold)
    {

        // cout<<"After space demanded: "<< allocated_size_maxId_fill <<"  " << "available size: "<< maxId_fill_size_threshhold << endl;
        break;
    }
    else
    {
        // cout<<"After space demanded: "<< allocated_size_maxId_fill <<"  "<< "available size: "<< maxId_fill_size_threshhold << endl;
        // N_src_group /= 2;
        N_src_group--;
        end_group = begin_group + N_src_group * N_gpu;	
        // cout<<"New reduced N_src_group: "<<N_src_group<<endl;
    }
}
// allocated_size_max_id=space_demanded;
allocated_size_max_id = (first_term * (ull_t)N_src_group)/2;
return N_src_group;

}

int_t Get_Max_id_efficent_N_SourceGroup_with_logical_bug(int_t N_src_group,int_t end_group, int_t N_gpu, int_t begin_group,
    ull_t maxId_fill_size_threshhold, ull_t& allocated_size_max_id,int_t vert_count)
{
// int_t groupsizefixed=0;
//Formula
// sum = n(a1+an)/2 where a1 = first term, an=last term and n is number of elements
//~Formula
// Total 0.25 GB= 67,108,864
N_src_group = 257;
cout<<"N_src_group changed: "<<N_src_group<<endl;
unsigned int first_term= (begin_group+1+N_gpu) + (end_group+1);
ull_t allocated_size_maxId_fill;
//here a1 =begin_group+1+N_gpu the #max_ids for the first source of the last GPU.
//and an=end_group+1 the #max_ids for the last source of the last GPU
while (1)
{
     allocated_size_maxId_fill=(first_term * (ull_t)N_src_group)/2 + N_src_group *vert_count; //space of both max + fill is allowed to push upto 90% of total memory
    // ull_t before_space= (unsigned)N_src_group * (unsigned)(end_group);
    // cout<<"Before space demanded "<< before_space<<endl;
    // if ((unsigned)N_src_group * (unsigned)(end_group) <= allocated_size)

    // max_id_size_threshhold

    // if (( first_term * (ull_t)N_src_group)/2 <= allocated_size)
    if ( allocated_size_maxId_fill <= maxId_fill_size_threshhold)
    {

        // cout<<"After space demanded: "<< allocated_size_maxId_fill <<"  " << "available size: "<< maxId_fill_size_threshhold << endl;
        break;
    }
    else
    {
        // cout<<"After space demanded: "<< allocated_size_maxId_fill <<"  "<< "available size: "<< maxId_fill_size_threshhold << endl;
        // N_src_group /= 2;
        N_src_group--;
        // cout<<"New reduced N_src_group: "<<N_src_group<<endl;
    }
}
// allocated_size_max_id=space_demanded;
allocated_size_max_id = (first_term * (ull_t)N_src_group)/2;
return N_src_group;

}

void symbfact_min_id(int args,char** argv,int myrank,ull_t& fill_count, double& time) 
//int main(int args, char **argv)
{
    std::cout<<"Input: ./exe beg csr weight #Processes chunk_size percent_cat0 percent_cat1 N_blocks_source_cat2 N_GPU_Node Mem_alloc_factor(5data)\n";
    if(args!=11){std::cout<<"Wrong input\n";exit(1);}

    const char *beg_file=argv[1];
    const char *end_file=argv[2];
    const char *csr_file=argv[3];
    // int num_process=atoi(argv[4]);
    int N_virtualGPU = atoi(argv[4]);//For Virtual GPU
    int chunk_size=atoi(argv[5]);
    int percent_cat0=atoi(argv[6]);
    int percent_cat2=atoi(argv[7]);
    int  N_blocks_source_cat2 = atoi(argv[8]);
    int N_GPU_Node=atoi(argv[9]);
    float total_memory_allocation_in_GB = atof(argv[10]);
    //int N_src_group=atoi(argv[10]);
   //For Virtual GPU
   double* Time = new double [N_virtualGPU];
   int_t* All_N_dumping_CPU_memory = new int_t [N_virtualGPU];
   double* TransferTime = new double [N_virtualGPU];
   ull_t* Edge_check_GPU = new ull_t [N_virtualGPU];
   ull_t* SIZE_COPIED = new ull_t [N_virtualGPU];
   for (int i=0;i<N_virtualGPU;i++)
   {
       Edge_check_GPU[i] = 0;
   }
   //~For Virtual GPU
    printf("My rank:%d\n",myrank);
    cout<<"Device number: "<<myrank % N_GPU_Node<<endl;
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


    // int_t sources_per_process=vert_count/num_process;
    int_t N_chunks= (int) ceil (vert_count/(float)chunk_size);

    // int temp_rank=myrank;
    // time=0;
    // if (temp_rank==0) cout<<"N_chunks: "<<N_chunks<<endl;

    // int* next_source_d;
    // H_ERR(cudaMalloc((void**) &next_source_d,sizeof(int_t))); 

    int_t N_section=3;



    int_t* frontier_size_d;
    int N_src_group=  Compute_Src_group_original(vert_count);//Target for larger N_src_group
    // int fill_N_src_group = Compute_Src_group(vert_count,total_memory_allocation_in_GB);
    // N_src_group=1024;//Forcing to be 128 for byte addressing testing
    // N_src_group=2048;//Forcing to be 128 for byte addressing testing
    //  N_src_group=4096;//Forcing to be 128 for byte addressing testing
    //  N_src_group=3072;
    //  N_src_group=2560;
    //  N_src_group=128;
    // N_src_group=256;
    // N_src_group=8192;//Forcing to be 128 for byte addressing testing
    cout<<"N_src_group: "<<N_src_group<<endl;
    // N_src_group=256;
    // cout<<"fill_N_src_group: "<<fill_N_src_group<<endl;

    cout<<"Allocating space for all 5 major datastructures: "<<endl;
    //***************************//
    // 1: max_id
    // 2: FQ
    // 3: Next FQ
    // 4: source_track_array
    // 5: next_source_trackarray
    //***************************//

    // ull_t Total_Memory_Allocation= total_memory_allocation_factor * 5*(ull_t)N_src_group*(ull_t)vert_count;
    // ull_t Total_Memory_Allocation= 5*(ull_t)fill_N_src_group*(ull_t)vert_count;
    ull_t Total_Memory_Allocation = (total_memory_allocation_in_GB*1024*1024*1024)/4;//For all 6 datastructures
    uint_t* Memory_allocated_d;
    H_ERR(cudaMalloc((void**) &Memory_allocated_d,sizeof(uint_t)*Total_Memory_Allocation)); 
    H_ERR(cudaMemset(Memory_allocated_d, 0, sizeof(uint_t)*Total_Memory_Allocation));//Setting fill-array to 0
    ull_t maxId_fill_size_threshhold= max_id_fraction * Total_Memory_Allocation;
    // ull_t max_id_size_threshhold= N_src_group * vert_count;//Need for initilization in worst case space requiorement of maxId[]


    int_t original_N_src_group=N_src_group;

    // H_ERR(cudaMalloc((void**) &frontier_size_d,sizeof(int_t)*N_src_group)); 
    // H_ERR(cudaMemset(frontier_size_d, 0, sizeof(int_t)*N_src_group));
    H_ERR(cudaMalloc((void**) &frontier_size_d,sizeof(int_t))); 
    H_ERR(cudaMemset(frontier_size_d, 0, sizeof(int_t)));


    // int_t* next_frontier_size_d;
    // // H_ERR(cudaMalloc((void**) &next_frontier_size_d,sizeof(int_t)*N_src_group)); 
    // // H_ERR(cudaMemset(next_frontier_size_d, 0, sizeof(int_t)*N_src_group));
    // H_ERR(cudaMalloc((void**) &next_frontier_size_d,sizeof(int_t))); 
    // H_ERR(cudaMemset(next_frontier_size_d, 0, sizeof(int_t)));

    // int_t* temp_next_frontier_size_d;
    // // H_ERR(cudaMalloc((void**) &next_frontier_size_d,sizeof(int_t)*N_src_group)); 
    // // H_ERR(cudaMemset(next_frontier_size_d, 0, sizeof(int_t)*N_src_group));
    // H_ERR(cudaMalloc((void**) &temp_next_frontier_size_d,sizeof(int_t))); 
    // H_ERR(cudaMemset(temp_next_frontier_size_d, 0, sizeof(int_t)));
    cout<<"Checking total and free GPU memory "<<endl;


    // ull_t optimium_concurrent_src_memory = (uint_t)N_src_group*(uint_t)vert_count;
    // uint_t max_id_space=(max_id_space_allocation*total_memory);//#bytes for max_id
    // ull_t max_id_space=(max_id_space_allocation*optimium_concurrent_src_memory);//#bytes for max_id


    // ull_t N_elements_max_id = max_id_space;

    // ull_t N_elements_max_id = (ull_t)N_src_group*(ull_t)vert_count;


    // cout<<"Initial max_id: "<<max_id_space<<endl;
    // cout<<"log2 of the max_id_space "<< log2((double)max_id_space) <<endl;

    // max_id_space=pow(2,(int)log2((double)max_id_space));

    // cout<<"#max_id_allocated "<<max_id_space<<" ("<<max_id_space/((double)1024*1024*1024)<< " GB /"<<(max_id_space/(double)total_memory)*100<<" %)"<<endl;
    // ull_t N_elements_max_id=max_id_space/4;
    // cout<< "N_elements_max_id allocated: " << N_elements_max_id<<endl;


    uint_t* cost_array_d;
    // H_ERR(cudaMalloc((void**) &cost_array_d,sizeof(uint_t)*N_elements_max_id));



    uint_t* fill_in_d;
    // H_ERR(cudaMalloc((void**) &fill_in_d,sizeof(uint_t)*N_src_group*vert_count)); 
    // H_ERR(cudaMemset(fill_in_d, 0, sizeof(uint_t)*N_src_group*vert_count));

    // ull_t ideal_allocated_frontier_size=(ull_t)N_src_group*(ull_t)vert_count;
    // ull_t allocated_frontier_size= ideal_allocated_frontier_size*FQspace_allocation;
    // ull_t real_allocation=allocated_frontier_size *real_allocation_factor;
    // cout<<"Allocated frontier size: "<<allocated_frontier_size<<endl;
    // cout<<"Real allocated frontier size: "<<real_allocation<<endl;
    uint_t* frontier_d;
    // H_ERR(cudaMalloc((void**) &frontier_d,sizeof(uint_t)*N_src_group*vert_count*frontier_multiple)); 
    // H_ERR(cudaMalloc((void**) &frontier_d,sizeof(int_t)*real_allocation)); 

    int* frontierchecked;
    // H_ERR(cudaMalloc((void**) &frontierchecked,sizeof(int_t)*N_src_group*vert_count*frontier_multiple)); 
    // H_ERR(cudaMemset(frontierchecked, 0, sizeof(int_t)*N_src_group*vert_count*frontier_multiple));


    uint_t* next_frontier_d;
    // H_ERR(cudaMalloc((void**) &next_frontier_d,sizeof(uint_t)*N_src_group*vert_count*frontier_multiple)); 
    // H_ERR(cudaMalloc((void**) &next_frontier_d,sizeof(int_t)*real_allocation)); 

    // unsigned int* max_frontier_size_d;
    // H_ERR(cudaMalloc((void**) &max_frontier_size_d,sizeof(unsigned int))); 
    // H_ERR(cudaMemset(max_frontier_size_d, 0, sizeof(unsigned int)));


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

    // int div_factor=ceil(log_2(N_src_group)/(double)8);
    // cout<<"div_factor: "<<div_factor<<endl;
    uint8_t* src_frontier_uint8_d, *next_src_frontier_uint8_d;
    uint16_t *src_frontier_uint16_d, *next_src_frontier_uint16_d;
    uint_t* src_frontier_d, *next_src_frontier_d;

    // H_ERR(cudaMalloc((void**) &src_frontier_d,sizeof(uint_t)*N_src_group*vert_count*frontier_multiple)); 
    // H_ERR(cudaMalloc((void**) &next_src_frontier_d,sizeof(uint_t)*N_src_group*vert_count*frontier_multiple)); 


    // H_ERR(cudaMalloc((void**) &src_frontier_d,sizeof(int_t)*real_allocation));  //stores the code (mapping) of source not the source itself
    // H_ERR(cudaMalloc((void**) &next_src_frontier_d,sizeof(int_t)*real_allocation)); 

    // switch(div_factor)
    // {   
    //     case 2: N_elements_max_idracking seed of traversal!"<<endl;
    //         H_ERR(cudaMalloc((void**) &src_frontier_uint16_d,sizeof(uint16_t)*N_src_group*vert_count*frontier_multiple)); 
    //         H_ERR(cudaMalloc((void**) &next_src_frontier_uint16_d,sizeof(uint16_t)*N_src_group*vert_count*frontier_multiple)); 
    //         break;

    //     default:
    //         //0<N_src_group<2147483648
    //         //Less likely for this condition   
    //         cout<<"4 bytes used for tracking seed of traversal!"<<endl;
    //         H_ERR(cudaMalloc((void**) &src_frontier_d,sizeof(int_t)*N_src_group*vert_count*frontier_multiple/div_factor));  
    //         H_ERR(cudaMalloc((void**) &next_src_frontier_d,sizeof(int_t)*N_src_group*vert_count*frontier_multiple/div_factor)); 

    // }

    int* source_d;
    H_ERR(cudaMalloc((void**) &source_d,sizeof(int_t)*N_src_group));
    int cfg_blockSize=128; 
    int BLKS_NUM,blockSize;
    cudaOccupancyMaxPotentialBlockSize( &BLKS_NUM, &blockSize, Compute_fillins_joint_traversal_group_wise_level_traversal, 0, 0);
    H_ERR(cudaDeviceSynchronize());
    BLKS_NUM = (blockSize * BLKS_NUM)/cfg_blockSize;
    int exp=log2((float)BLKS_NUM);
    BLKS_NUM=pow(2,exp);
    blockSize = cfg_blockSize;
#ifdef overwrite_kernel_config
    BLKS_NUM=512;  
    blockSize=128;
    // BLKS_NUM=1;  
    // blockSize=32;
    // blockSize=2;
#endif
    int_t* my_current_frontier_d;
    H_ERR(cudaMalloc((void**) &my_current_frontier_d,sizeof(int_t)*BLKS_NUM*blockSize));  
    //BLKS_NUM=128; //Currently implemented for 128 blocks



    cout<<"Detected GridDim: "<<BLKS_NUM<<endl;

    int* lock_d;
    H_ERR(cudaMalloc((void**) &lock_d,sizeof(int)*BLKS_NUM)); //size of lock_d is num of blocks
    H_ERR(cudaMemset(lock_d, 0, sizeof(int)*BLKS_NUM));
    H_ERR(cudaThreadSynchronize());

    int_t N_groups=(ceil) (vert_count/(float)N_src_group);
    Barrier global_barrier(BLKS_NUM); 
    cout<<"BLKS_NUM: "<<BLKS_NUM<<endl;
    cout<<"blockSize: "<<blockSize<<endl;




    int_t* next_front_d;
    H_ERR(cudaMalloc((void**) &next_front_d,sizeof( int_t))); //size of lock_d is num of blocks
    // int_t next_front=BLKS_NUM*blockSize;
    int_t Nwarp = (BLKS_NUM*blockSize) >> 5;
    int_t next_front;
    #ifdef thread_centric
    next_front = (BLKS_NUM*blockSize); 
    #else
     next_front = Nwarp; //For warp-centric version
    #endif
    H_ERR(cudaMemcpy(next_front_d,&next_front,sizeof(int_t),cudaMemcpyHostToDevice));

    int_t* count_thread_exiting;
    H_ERR(cudaMalloc((void**) &count_thread_exiting,sizeof( int_t))); //size of lock_d is num of blocks
    H_ERR(cudaMemset(count_thread_exiting,0,sizeof( int_t)));

    //Allocate buffer memory for frontiers in CPU
    uint_t* CPU_buffer_next_level;//= (int*) malloc (N_src_group*vert_count*sizeof(int_t));
    H_ERR(cudaMallocHost((void**)&CPU_buffer_next_level,N_src_group*vert_count*sizeof(uint_t)));
    // H_ERR(cudaHostAlloc((void**)&CPU_buffer_next_level,N_src_group*vert_count*sizeof(int_t),cudaHostAllocDefault));

    uint_t* CPU_buffer_next_level_source;//= (int*) malloc (N_src_group*vert_count*sizeof(int_t));

    H_ERR(cudaMallocHost((void**)&CPU_buffer_next_level_source,N_src_group*vert_count*sizeof(uint_t)));
    // H_ERR(cudaHostAlloc((void**)&CPU_buffer_next_level_source,N_src_group*vert_count*sizeof(int_t),cudaHostAllocDefault));
    int_t size_CPU_buffer_next_level=0;

    uint_t* CPU_buffer_current_level;//= (int*) malloc (N_src_group*vert_count*sizeof(int_t));
    H_ERR(cudaMallocHost((void**)&CPU_buffer_current_level,N_src_group*vert_count*sizeof(uint_t)));
    // H_ERR(cudaHostAlloc((void**)&CPU_buffer_current_level,N_src_group*vert_count*sizeof(int_t),cudaHostAllocDefault));

    uint_t* CPU_buffer_current_level_source;//= (int*) malloc (N_src_group*vert_count*sizeof(int_t));
    H_ERR(cudaMallocHost((void**)&CPU_buffer_current_level_source,N_src_group*vert_count*sizeof(uint_t)));
    // H_ERR(cudaHostAlloc((void**)&CPU_buffer_current_level_source,N_src_group*vert_count*sizeof(int_t),cudaHostAllocDefault));

    int_t size_CPU_buffer_current_level=0;

    int_t* next_N_buffer_CPU=(int*) malloc (10000*sizeof(int_t));
    int_t* current_N_buffer_CPU=(int*) malloc (10000*sizeof(int_t));

    int_t* swap_CPU_buffers_m;
    H_ERR(cudaMallocManaged((void**) &swap_CPU_buffers_m,sizeof( int_t))); 
    H_ERR(cudaMemset(swap_CPU_buffers_m,0,sizeof( int_t)));

    int_t* swap_GPU_buffers_m;
    H_ERR(cudaMallocManaged((void**) &swap_GPU_buffers_m,sizeof( int_t))); 
    H_ERR(cudaMemset(swap_GPU_buffers_m,0,sizeof( int_t)));

    int_t* dump_m;
    int_t* load_m;
    int_t* current_buffer_m;
    int_t* next_buffer_m;
    H_ERR(cudaMallocManaged((void**) &next_buffer_m,sizeof( int_t))); 
    H_ERR(cudaMallocManaged((void**) &current_buffer_m,sizeof( int_t))); 
    current_buffer_m[0]=0;
    next_buffer_m[0]=0;
    H_ERR(cudaMallocManaged((void**) &dump_m,sizeof( int_t))); 
    // H_ERR(cudaMalloc((void**) &dump_m,sizeof( int_t))); 
    H_ERR(cudaMemset(dump_m,0,sizeof( int_t)));
    // H_ERR(cudaMalloc((void**) &load_m,sizeof( int_t))); 
    H_ERR(cudaMallocManaged((void**) &load_m,sizeof( int_t))); 
    H_ERR(cudaMemset(load_m,0,sizeof( int_t)));
    // int_t buffer_flag_host=0;
    // int_t buffer_flag_getfronter_host=0;

    int_t* offset_next_kernel;
    H_ERR(cudaMallocManaged((void**) &offset_next_kernel,sizeof( int_t))); 
    // H_ERR(cudaMemset(offset_next_kernel,0,sizeof( int_t)));
    // offset_next_kernel[0]= INT_MAX;
    offset_next_kernel[0]= 0;
    int_t* offset_kernel;
    H_ERR(cudaMallocManaged((void**) &offset_kernel,sizeof( int_t))); 
    // H_ERR(cudaMemset(offset_next_kernel,0,sizeof( int_t)));
    offset_kernel[0]= 0;

    uint_t max_id_offset = MAX_VAL-vert_count;//vert_count*group;
    uint_t group_MAX_VAL = max_id_offset + vert_count;
    int_t reinitialize=1;
    uint_t group_loops=MAX_VAL/vert_count;
    uint_t count_group_loop=0;
    int_t N_dumping_cpu_memory=0;
    int_t N_reading_cpu_memory=0;

    int_t* next_frontier_size_d;
    H_ERR(cudaMalloc((void**) &next_frontier_size_d,sizeof(int_t))); 
    H_ERR(cudaMemset(next_frontier_size_d, 0, sizeof(int_t)));

    int_t* remaining_frontier_size_d;
    H_ERR(cudaMalloc((void**) &remaining_frontier_size_d,sizeof(int_t))); 

    // H_ERR(cudaMalloc((void**) &next_frontier_size_d,sizeof(int_t)*N_src_group)); 
    // H_ERR(cudaMemset(next_frontier_size_d, 0, sizeof(int_t)*N_src_group));


    double start_time;
    double dumping_loading_time=0;
    ull_t last_fill_count=0;

    ull_t size_copied=0;
    double start_only_Copy=0;
    double time_only_Copy=0;

    size_t free_memory,total_memory;
    H_ERR(cudaMemGetInfo(&free_memory, &total_memory));

    // cout<<"The total memory in GPU: "<<total_memory/((double)1024*1024*1024) <<" GB"<<endl;
    // cout<<"The free memory in GPU: "<<free_memory/((double)1024*1024*1024)<<" GB"<<endl; 
    // cout<<endl;
    // cout<<"Running the merge kernel"<<endl;

    // ull_t fill_count=0;
    // switch(div_factor)
    // {   
    //     case 2: 
    //         start_time=wtime();
    //         //   double start_time_first=wtime();

    //         //Compute_fillins_merge_traverse <<<BLKS_NUM,blockSize>>> (cost_array_d,fill_in_d,frontier_d,next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,myrank,num_process,src_frontier_d,next_src_frontier_d,source_d, frontier_size_d, next_frontier_size_d,lock_d,N_groups, atomic_time_all_threads_d,average_frontier_size_d,total_edge_check_d,next_front_d,max_id_clock_cycle_d,symb_fact_clock_cycle_d);
    //         // Compute_fillins_merge_traverse_multiGPU_uint8 <<<BLKS_NUM,blockSize>>> (cost_array_d,fill_in_d,frontier_d,
    //         //         next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,myrank,num_process,
    //         //         src_frontier_uint8_d,next_src_frontier_uint8_d,source_d, frontier_size_d, next_frontier_size_d,
    //         //         lock_d,N_groups, atomic_time_all_threads_d,average_frontier_size_d,total_edge_check_d,
    //         //         next_front_d,max_id_clock_cycle_d,symb_fact_clock_cycle_d,N_src_group,max_frontier_size_d,
    //         //         max_re_inserted_vertex_d,average_re_inserted_vertex,
    //         //         duplicate_insertions_same_src_curr_front,
    //         //         average_reinsrtion_group_d);    
    //         // H_ERR(cudaDeviceSynchronize()); 
    //         break;
    //     case 1:
    int_t begin_group=0;
    // begin_group = 3840;
    int_t loop=0;
    ull_t last_allocated_size_max_id=0;
    int Total_dumpings =0;
    ull_t Total_fill_count=0;
    ull_t minimum_out_core_datasize=Total_Memory_Allocation;
    ull_t maximum_out_core_datasize=0;
    int beginning_C =0 ;
    ull_t minimul4_out_core_allocation=UINT_MAX;
    maximum_front_size=0;
    int max_dumping=0;
    int max_dumping_gpu=0;
    for (int GPU_ID=0; GPU_ID < N_virtualGPU; GPU_ID++)
    {
        // int_t N_no_init = (4294967295/vert_count); //(2^32-1/|V|)
        // int_t no_init_cnt=0;
        bool start = true;
       
        myrank = GPU_ID;
        int_t num_process = N_virtualGPU;
       
    start_time=wtime();


    /*   Compute_fillins_merge_traverse_multiGPU_uint16 <<<BLKS_NUM,blockSize>>> (cost_array_d,fill_in_d,frontier_d,
         next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,myrank,num_process,
         src_frontier_uint16_d,next_src_frontier_uint16_d,source_d, frontier_size_d, next_frontier_size_d,
         lock_d,N_groups, atomic_time_all_threads_d,average_frontier_size_d,total_edge_check_d,
         next_front_d,max_id_clock_cycle_d,symb_fact_clock_cycle_d,N_src_group,max_frontier_size_d,
         max_re_inserted_vertex_d,average_re_inserted_vertex,
         duplicate_insertions_same_src_curr_front,
         average_reinsrtion_group_d); 
         H_ERR(cudaDeviceSynchronize()); 
     */

    // Initialise_cost_array<<<128,128>>>(cost_array_d,N_elements_max_id,group_MAX_VAL);
    // Initialise_cost_array<<<128,128>>>(Memory_allocated_d,N_elements_max_id,group_MAX_VAL);

    // Initialise_cost_array<<<128,128>>>(Memory_allocated_d,N_src_group*vert_count,group_MAX_VAL);
  
  
    // Initialise_cost_array<<<128,128>>>(Memory_allocated_d,maxId_fill_size_threshhold,group_MAX_VAL);
    // H_ERR(cudaDeviceSynchronize());
    

    // for (int_t group=0; group< N_groups;group+=num_process)//N_gpu=num_process
    // for (int_t group=5; group< 8; group+=num_process)//N_gpu=num_process
    // int_t group=7;
    //  int_t group=31;
    // for (int_t group=90; group< 91; group+=num_process)//N_gpu=num_process
    // if (begin_group < 4096)
    while(begin_group < vert_count)
    // begin_group = 1024;
    // while(begin_group < 512)
    {
        // cout<<endl<<endl;
        int_t max_vertex_group=begin_group + N_src_group * num_process;	
        // printf("myrank:%d loop:%d Before max_vertex_group:%d Before N_src_group:%d\n",myrank,loop,max_vertex_group,N_src_group);
        ull_t allocated_size_max_id;
        N_src_group = Get_Max_id_efficent_N_SourceGroup(N_src_group,max_vertex_group, num_process, begin_group,
                maxId_fill_size_threshhold,allocated_size_max_id,vert_count);
                // cout<<"Total_Memory_Allocation: "<< Total_Memory_Allocation<<endl;
                
        // cout<<"New N_src_group: "<<N_src_group<<endl;
        if (start) {beginning_C = N_src_group; start=false;}
        ull_t allocated_size_fill = N_src_group* vert_count;

        // cout<<"Allocated size_max_id: "<<allocated_size_max_id<<endl;
        // cout<<"allocated_size_fill: "<<allocated_size_fill<<endl;
        // cout<<"Allocations left for remaining datastructures: "<<(Total_Memory_Allocation-allocated_size_fill-allocated_size_max_id)<<endl;

        if (minimul4_out_core_allocation < (Total_Memory_Allocation-allocated_size_fill-allocated_size_max_id))
        {
            minimul4_out_core_allocation=(Total_Memory_Allocation-allocated_size_fill-allocated_size_max_id);
        }
        // ull_t allocated_size_fill = 0;
        if (last_allocated_size_max_id < allocated_size_max_id)
        {
            Initialise_cost_array<<<128,128>>>(&Memory_allocated_d[last_allocated_size_max_id],allocated_size_max_id-last_allocated_size_max_id,group_MAX_VAL);
            H_ERR(cudaDeviceSynchronize());
        }
        // if ((Total_Memory_Allocation- max_id_size_threshhold) < minimum_out_core_datasize) 
        // {
        //     minimum_out_core_datasize= Total_Memory_Allocation- max_id_size_threshhold;
        // }
        // if ((Total_Memory_Allocation- max_id_size_threshhold) > maximum_out_core_datasize) 
        // {
        //     maximum_out_core_datasize = Total_Memory_Allocation- max_id_size_threshhold;
        // }
        last_allocated_size_max_id=allocated_size_max_id;
        //Dividing the memory structures among the data structures
        // cout<<"max_id_size_threshhold: "<<max_id_size_threshhold<<endl;
        // cout<<"Allocated size for max_id: "<<allocated_size_max_id<<endl;
        // cout<<"Total_Memory_Allocation: "<<Total_Memory_Allocation<<endl;

        // ull_t real_allocation= (Total_Memory_Allocation - allocated_size_max_id)/4; // Division among 4 major data-structures
        ull_t real_allocation= (Total_Memory_Allocation - (allocated_size_max_id+allocated_size_fill))/4; // Division among 4 major data-structures
        // ull_t real_allocation= 50000; // Division among 4 major data-structures

        // ull_t real_allocation = 65536; 

        ull_t allocated_frontier_size= real_allocation/real_allocation_factor;
        // cout<<"real_allocation: "<<real_allocation<<endl;
        // cout<<"allocated_frontier_size: "<<allocated_frontier_size<<endl;
        // cout<<"N_src_group*vert_count: "<<N_src_group*vert_count<<endl;
        //marking the territories

        // cost_array_d=Memory_allocated_d;
        // H_ERR(cudaMalloc((void**) &cost_array_d,sizeof(uint_t)*(allocated_size_max_id))); 
        // Initialise_cost_array<<<128,128>>>(cost_array_d,allocated_size_max_id,group_MAX_VAL);
        // H_ERR(cudaDeviceSynchronize());

        // H_ERR(cudaMalloc((void**) &cost_array_d,sizeof(uint_t)*N_src_group*vert_count)); 
        // frontier_d = &Memory_allocated_d[allocated_size_max_id];
        // frontier_d = &Memory_allocated_d[allocated_size_max_id+real_allocation*2];//+ (uint_t)allocated_size_max_id + (uint_t) real_allocation;
        // H_ERR(cudaMalloc((void**) &frontier_d,sizeof(uint_t)*real_allocation)); 
        // Initialise_cost_array<<<128,128>>>(Memory_allocated_d,Total_Memory_Allocation,group_MAX_VAL);
        // H_ERR(cudaDeviceSynchronize());

        // Initialise_Numbers<<<128,128>>>(Memory_allocated_d,Total_Memory_Allocation);
        // H_ERR(cudaDeviceSynchronize());

        cost_array_d=Memory_allocated_d; 
        // cout<<"N_src_group: "<<N_src_group<<endl;
        

       
        // ull_t frontier_offset=allocated_size_max_id;
        ull_t frontier_offset=allocated_size_max_id;

        ull_t next_frontier_offset=allocated_size_max_id+real_allocation;
        ull_t src_track_array_offset=allocated_size_max_id+2*real_allocation;
        ull_t next_src_track_array_offset=allocated_size_max_id+3*real_allocation;
        ull_t fill_array_offset =allocated_size_max_id+4*real_allocation;
        // cout<<"maxId[] range: 0 to "<<fill_array_offset<<endl;
        // cout<<"fill[] range:"<< fill_array_offset <<" to" <<frontier_offset  <<endl;
        // cout<<"frontierQueue[] range:"<< frontier_offset <<" to" <<next_frontier_offset  <<endl;
        // cout<<"NewfrontierQueue[] range:"<< next_frontier_offset <<" to" <<src_track_array_offset  <<endl;
        // cout<<"tracker[] range:"<< src_track_array_offset <<" to" <<next_src_track_array_offset  <<endl;
        // cout<<"NewTracker[] range:"<< next_src_track_array_offset <<" to" <<next_src_track_array_offset + real_allocation <<endl;
        
        assert(allocated_size_max_id+allocated_size_fill+4*real_allocation <= Total_Memory_Allocation);

        fill_in_d = Memory_allocated_d + fill_array_offset;
        // H_ERR(cudaMemset(fill_in_d, 0, sizeof(uint_t)*N_src_group*vert_count));
        frontier_d = Memory_allocated_d+ frontier_offset;// + allocated_size_max_id;

        next_frontier_d = Memory_allocated_d + next_frontier_offset;
        src_frontier_d = Memory_allocated_d + src_track_array_offset;
        next_src_frontier_d = Memory_allocated_d + next_src_track_array_offset;

        //~marking the territories

#ifdef enable_debug
        if (allocated_frontier_size > INT_MAX) cout<<"Warning!: Allocated frontier size exceesed INT range (2 GB)"<<endl;
#endif

        // int_t  temp_allocated_frontier_size=allocated_frontier_size;
        //~Dividing the memory structures among the data structures
        max_vertex_group = begin_group + N_src_group * num_process;	
        // max_vertex_group= (max_vertex_group > vert_count) ? vert_count:max_vertex_group;
        int_t first_source=begin_group+myrank;//optimization for removing the intermediate bubbles in the cost array

        // cout<<"Max_id dedicated #source per group: "<<N_src_group<<endl;
        // cout<<"allocated size: "<< allocated_frontier_size<<endl;
        // cout<<"Real allocation size for frontier: "<<real_allocation<<endl;

        // cout<<"Max_vertex in group + 1: "<<max_vertex_group<<endl;

        // printf("myrank:%d loop:%d begin_group:%d N_src_group:%d max_vertex_group:%d\n",myrank,loop,begin_group,N_src_group,max_vertex_group);

        swap_GPU_buffers_m[0]=0;

        // H_ERR(cudaMemcpy(remaining_frontier_size_d,&temp_allocated_frontier_size,sizeof(int_t),cudaMemcpyHostToDevice));
        Compute_fillins_joint_traversal_group_wise_splitted (cost_array_d,fill_in_d,frontier_d,
                next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,myrank,num_process,
                src_frontier_d,next_src_frontier_d,source_d, frontier_size_d, next_frontier_size_d,
                lock_d,N_groups,
                dump_m,load_m,
                N_src_group,/*group,*/max_id_offset,next_front_d,
                allocated_frontier_size,my_current_frontier_d,frontierchecked,
                swap_GPU_buffers_m,count_thread_exiting,BLKS_NUM,blockSize,
                remaining_frontier_size_d,begin_group, max_vertex_group,allocated_size_max_id,first_source,
                real_allocation,Memory_allocated_d,Nwarp); 
        // cout<<"finished a global gropu: "<< begin_group <<endl;


        while ((dump_m[0]==1) || (load_m[0] ==1))
        {
#ifdef profile_dumping_loading_time
            double start_temp_dump_loadtime=wtime();
#endif
            // cout<<"Either dumping or loading with begin group: "<< begin_group <<endl;
#ifdef enable_debug
            cout<<"Either dumping or loading!"<<endl;


            H_ERR(cudaMemcpy(&fill_count,fill_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
            cout<<"#fill count till now = "<<fill_count-last_fill_count<<endl;
#endif

            if (swap_CPU_buffers_m[0]==1)
            {
#ifdef enable_debug
                cout<<"swapping current and next CPU buffers!"<<endl;
#endif
                SWAP_CPU_BUFFERS(CPU_buffer_next_level,
                        CPU_buffer_next_level_source,
                        size_CPU_buffer_next_level,
                        CPU_buffer_current_level,CPU_buffer_current_level_source,size_CPU_buffer_current_level,current_N_buffer_CPU,
                        next_N_buffer_CPU,swap_CPU_buffers_m);
#ifdef enable_debug
                cout<<"current_N_buffer_CPU["<<current_buffer_m[0]-1<<"]="<<current_N_buffer_CPU[current_buffer_m[0]-1]<<endl;
#endif
            }

            if (swap_GPU_buffers_m[0]==1)
                // if (false)
            {
#ifdef enable_debug
                cout<<"swapping current and next FQs in globally !"<<endl;
#endif
                SWAP_GPU_BUFFERS(frontier_d,next_frontier_d,src_frontier_d,next_src_frontier_d,swap_GPU_buffers_m);
            }

            if (dump_m[0]==1)
            {             
                //copy next front from GPU to CPU memory

                int_t temp_front_size;                
                H_ERR(cudaMemcpy(&temp_front_size,next_frontier_size_d,sizeof(int_t),cudaMemcpyDeviceToHost));
#ifdef enable_debug
                cout<<"Dumping to the CPU memory!"<<endl;
                // cout<<"#frontiers dumped into the CPU memory: "<<temp_front_size<<endl;
                // cout<<"#frontiers dumped into the CPU memory: "<<temp_front_size<<endl;
                // cout<<"4*N_src_group*vert_counT: "<< N_src_group*vert_count<<endl;
                // cout<<"size_CPU_buffer_next_level "<< size_CPU_buffer_next_level<<endl;
                // cout<<"allocated size: "<< allocated_frontier_size<<endl;

                // cout<<"Amount of CPU buffer remaining:  "<<N_src_group*vert_count-size_CPU_buffer_next_level<<endl;
#endif
                //  assert (temp_front_size <= temp_allocated_frontier_size);


                //  swap_ptr_index(next_frontier_size_d, temp_next_frontier_size_d);
                H_ERR(cudaMemset(next_frontier_size_d, 0, sizeof(int_t)));


                // cout<<"#frontiers dumped into the CPU memory: "<<temp_front_size<<endl;
                // cout<<"N_src_group*vert_counT: "<< N_src_group*vert_count<<endl;
                // cout<<"size_CPU_buffer_next_level "<< size_CPU_buffer_next_level<<endl;
                // cout<<"allocated size: "<< allocated_frontier_size<<endl;
                // cout<<"real allocated size: "<< real_allocation << endl;
                // cout<<"Dumping"<<endl;
#ifdef profile_memcpy_bandwidth
                start_only_Copy=wtime();
#endif

                // cout<<"dumped frontier size: "<<temp_front_size<<endl;
                // cout<<"allocated size: "<< allocated_frontier_size<<endl;
                // cout<<"Real allocation size for frontier: "<<real_allocation<<endl;

                assert(temp_front_size <= real_allocation);
                H_ERR(cudaMemcpy(&CPU_buffer_next_level[size_CPU_buffer_next_level],next_frontier_d,sizeof(int_t)*temp_front_size,cudaMemcpyDeviceToHost));

                H_ERR(cudaMemcpy(&CPU_buffer_next_level_source[size_CPU_buffer_next_level],next_src_frontier_d,sizeof(int_t)*temp_front_size,cudaMemcpyDeviceToHost));
                // H_ERR(cudaMemcpy(remaining_frontier_size_d,&temp_allocated_frontier_size,sizeof(int_t),cudaMemcpyHostToDevice));
#ifdef profile_memcpy_bandwidth
                time_only_Copy += (wtime()-start_only_Copy);
#endif
                next_N_buffer_CPU[next_buffer_m[0]]=temp_front_size;
#ifdef enable_debug
                cout<<"next_N_buffer_CPU["<<next_buffer_m[0]<<"]="<<next_N_buffer_CPU[next_buffer_m[0]]<<endl;
#endif

                size_CPU_buffer_next_level += temp_front_size;
#ifdef enable_debug
                // cout<<"Frontiers in CPU buffer after dumping"<<endl;
                // Display_CPU_buffers( size_CPU_buffer_next_level,  CPU_buffer_next_level,  CPU_buffer_next_level_source,0,allocated_frontier_size);
                cout<<endl;
#endif

#ifdef profile_memcpy_bandwidth
                size_copied+= (2*temp_front_size);
#endif
                H_ERR(cudaMemcpy(offset_kernel,next_front_d,sizeof(int_t),cudaMemcpyDeviceToDevice));
                // H_ERR(cudaMemset(dump_m,0,sizeof( int_t)));
                next_buffer_m[0]++;
                dump_m[0]=0;
                N_dumping_cpu_memory++;
#ifdef enable_debug
                // cout<<"Before auxiliary kernel offset_kernel[0]= "<< offset_kernel[0]<<endl;
                // cout<<"current_buffer_m: "<<current_buffer_m[0]<<endl;
                // cout<<"next_buffer_m: "<<next_buffer_m[0]<<endl;
#endif

                //   offset_next_kernel[0]=INT_MAX;
                //   offset_next_kernel[0]=0;
            }
            if (load_m[0] == 1)
            {
                //copy from CPU memory to the GPU memory
                // int_t temp_front_size;                
                // H_ERR(cudaMemcpy(&temp_front_size,next_frontier_size_d,sizeof(int_t),cudaMemcpyDeviceToHost));
                // cout<<"Loading from CPU memory!"<<endl;
                current_buffer_m[0]--;

                int_t temp_front_size = current_N_buffer_CPU[current_buffer_m[0]];
#ifdef enable_debug
                // cout<<"current_buffer_m: "<<current_buffer_m[0]<<endl;
                // cout<<"Reading from the CPU memory!"<<endl;
                // cout<<"Frontiers in CPU buffer before loading"<<endl;
#endif
                // Display_CPU_buffers(size_CPU_buffer_current_level,CPU_buffer_current_level,CPU_buffer_current_level_source,1,allocated_frontier_size);
                size_CPU_buffer_current_level -= temp_front_size;
#ifdef enable_debug
                assert(size_CPU_buffer_current_level >= 0);
                cout<<"#read frontiers from the CPU memory: "<<temp_front_size<<endl;
#endif
#ifdef profile_memcpy_bandwidth
                start_only_Copy=wtime();
#endif
                H_ERR(cudaMemcpy(frontier_d, &CPU_buffer_current_level[size_CPU_buffer_current_level], sizeof(int_t)*temp_front_size,cudaMemcpyHostToDevice));                
                H_ERR(cudaMemcpy(src_frontier_d, &CPU_buffer_current_level_source[size_CPU_buffer_current_level], sizeof(int_t)*temp_front_size,cudaMemcpyHostToDevice));                
                H_ERR(cudaMemcpy(frontier_size_d,&temp_front_size,sizeof(int_t),cudaMemcpyHostToDevice));

#ifdef profile_memcpy_bandwidth
                time_only_Copy += (wtime()-start_only_Copy);
#endif

#ifdef enable_debug
                // cout<<"Frontiers in CPU buffer after loading"<<endl;
                // Display_CPU_buffers(size_CPU_buffer_current_level,CPU_buffer_current_level,CPU_buffer_current_level_source,1,allocated_frontier_size);
#endif
#ifdef profile_memcpy_bandwidth
                size_copied += (2*temp_front_size);
#endif
                // H_ERR(cudaMemset(load_m,0,sizeof( int_t)));
                // next_buffer_m[0]--;

                load_m[0]=0;
                // offset_next_kernel[0]=INT_MAX; // When loading the frontier the variables needs to be initialized to 0. So the threads start from the
                //beginning of the loaded frontier
                // offset_next_kernel[0]=0;
                // offset_kernel[0]=BLKS_NUM*blockSize;
                #ifdef thread_centric
                offset_kernel[0]=(BLKS_NUM*blockSize);
                #else
                offset_kernel[0]=(BLKS_NUM*blockSize) >> 5;
                #endif
                N_reading_cpu_memory++;

#ifdef enable_debug
                // cout<<"Number of reading from cpu memory till now: "<<N_reading_cpu_memory<<endl;
                // cout<<"current_buffer_m: "<<current_buffer_m[0]<<endl;
                // cout<<"next_buffer_m: "<<next_buffer_m[0]<<endl;
#endif

                // buffer_flag_getfronter_host=0;
            }

            //This kernel is repeatedly called until the current group traversal finishes. 
            // int_t temp_current_frontiers_size=0;
            // H_ERR(cudaMemcpy(&temp_current_frontiers_size,frontier_size_d,sizeof(int_t),cudaMemcpyDeviceToHost));
            // cout<<"Before Auxiliary kernel frontier_size: "<<temp_current_frontiers_size<<endl;
            H_ERR(cudaDeviceSynchronize());
            // test_kernel <<<BLKS_NUM,blockSize>>> (frontier_size_d, cost_array_d, fill_in_d, frontier_d,
            //     next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,
            //     src_frontier_d,next_src_frontier_d,source_d,  next_frontier_size_d,
            //     lock_d,
            //     dump_m,load_m,                           
            //     max_id_offset,
            //     next_front_d,allocated_frontier_size,my_current_frontier_d,current_buffer_m,next_buffer_m,offset_kernel,
            // group,swap_CPU_buffers_m); 

#ifdef profile_dumping_loading_time
            dumping_loading_time += (wtime()-start_temp_dump_loadtime);
#endif

            // Compute_fillins_Auxiliary <Compute_fillins_Auxiliary_splittednize());


            Compute_fillins_Auxiliary_splitted (frontier_size_d, cost_array_d, fill_in_d, frontier_d,
                    next_frontier_d,vert_count,csr_d,col_st_d,col_ed_d,fill_count_d,
                    src_frontier_d,next_src_frontier_d,source_d,  next_frontier_size_d,
                    lock_d,
                    dump_m,load_m,                           
                    max_id_offset,    
                    next_front_d,allocated_frontier_size,my_current_frontier_d,current_buffer_m,next_buffer_m,offset_kernel,
                    swap_CPU_buffers_m,frontierchecked,swap_GPU_buffers_m,BLKS_NUM,blockSize,
                    remaining_frontier_size_d,max_vertex_group,first_source,num_process,allocated_size_max_id,real_allocation,Nwarp); 


            //  offset_next_kernel[0]=INT_MAX;

            // }
            // H_ERR(cudaMemcpy(&buffer_flag_host,dump_m,sizeof(int_t),cudaMemcpyDeviceToHost));
    }
#ifdef enable_debug
    // cout<<"current_buffer_m: "<<current_buffer_m[0]<<endl;
    // cout<<"next_buffer_m: "<<next_buffer_m[0]<<endl;
    assert(current_buffer_m[0]==0);
    assert(next_buffer_m[0]==0);
#endif

    max_id_offset-=vert_count;
    count_group_loop++;
    if (count_group_loop >= group_loops)
        // if ((max_id_offset-vert_count) < 0)
    {
        max_id_offset=MAX_VAL-vert_count;           
        group_MAX_VAL = max_id_offset + vert_count;
        // cout<<"REINITIALIZATION OF COST ARRAY"<<endl;
        // Initialise_cost_array<<<128,128>>>(cost_array_d,max_id_size_threshhold,group_MAX_VAL);
        Initialise_cost_array<<<128,128>>>(cost_array_d,allocated_size_max_id,group_MAX_VAL);
        H_ERR(cudaDeviceSynchronize());
        count_group_loop=0;
    }
    else
    {
        group_MAX_VAL = max_id_offset + vert_count;
        // Initialise_cost_array<<<128,128>>>(&cost_array_d[allocated_size_max_id],max_id_size_threshhold-allocated_size_max_id,group_MAX_VAL);
        // H_ERR(cudaDeviceSynchronize());
        // cout<<"NEW GROUP_MAX_VAL: "<<group_MAX_VAL<<endl;
    }
    // last_allocated_size_max_id = allocated_size_max_id;
    // H_ERR(cudaMemcpy(&fill_count,fill_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
    // cout<<"#Fill-ins till now: "<<fill_count<<endl;
#ifdef enable_debug
    cout<<"#fillsins group: ="<<fill_count-last_fill_count<<endl;
    last_fill_count=fill_count;
#endif
    begin_group+=(N_src_group*num_process);
    loop++;
}


time =((wtime()-start_time)*1000);

Time[GPU_ID] = time;
//Reinitialize variables for another GPU
H_ERR(cudaMemcpy(next_front_d,&next_front,sizeof(int_t),cudaMemcpyHostToDevice));
H_ERR(cudaMemset(swap_CPU_buffers_m,0,sizeof( int_t)));
H_ERR(cudaMemset(swap_GPU_buffers_m,0,sizeof( int_t)));
current_buffer_m[0]=0;
next_buffer_m[0]=0;
H_ERR(cudaMemset(dump_m,0,sizeof( int_t)));
H_ERR(cudaMemset(load_m,0,sizeof( int_t)));
offset_next_kernel[0]= 0;
H_ERR(cudaMemset(next_frontier_size_d, 0, sizeof(int_t)));
// H_ERR(cudaMemset(fill_in_d, 0, sizeof(int_t)*N_src_group*vert_count));
H_ERR(cudaMemset(Memory_allocated_d, 0, sizeof(uint_t)*Total_Memory_Allocation));//setting fill_in[] array to 0
H_ERR(cudaMemset(frontier_size_d, 0, sizeof(int_t)));


TransferTime[GPU_ID] = dumping_loading_time;

All_N_dumping_CPU_memory[GPU_ID] = N_dumping_cpu_memory;

SIZE_COPIED[GPU_ID]=size_copied;
cout<<"Time for GPU "<<GPU_ID<<" :"<<Time[GPU_ID]<<" ms"<<endl;
if (max_dumping < N_dumping_cpu_memory) {max_dumping=N_dumping_cpu_memory; max_dumping_gpu=myrank;}
Total_dumpings+=N_dumping_cpu_memory;
H_ERR(cudaMemcpy(&fill_count,fill_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
Total_fill_count += fill_count;
N_dumping_cpu_memory=0;
dumping_loading_time=0;
max_id_offset = MAX_VAL-vert_count;//vert_count*group;
group_MAX_VAL = max_id_offset + vert_count;
reinitialize=1;
group_loops=MAX_VAL/vert_count;
count_group_loop=0;
N_dumping_cpu_memory=0;
N_reading_cpu_memory=0;
begin_group =0;
loop=0;
last_allocated_size_max_id=0;
//Reinitialize variables for another GPU
H_ERR(cudaMemset(fill_count_d, 0, sizeof(ull_t)));
}
cout<<"merge traversal complete!"<<endl;


double minimum_time = Time[0];
double maximum_time = Time[0];
int_t GPU_slowest_time =0;
int_t GPU_fastest_time =0;
int_t GPU_least_edgeChecks =0;
int_t GPU_highest_edgeChecks =0;
ull_t max_edge_checks = Edge_check_GPU[0];
ull_t min_edge_checks = Edge_check_GPU[0];
// ull_t max_size_copied = SIZE_COPIED[0];
// ull_t min_size_copied = SIZE_COPIED[0];

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
    // if (SIZE_COPIED[0][i] > max_size_copied) 
    // {
    //     max_size_copied = SIZE_COPIED[0][i];
    //     GPU_highest_edgeChecks = i;
    // }
    // if (Edge_check_GPU[i] < min_edge_checks) 
    // {
    //     min_edge_checks = Edge_check_GPU[i];
    //     GPU_least_edgeChecks = i;
    // }
}


// cout<<"N_elements_max_id: "<<N_elements_max_id<<"  original_N_src_group: "<<original_N_src_group<<endl;
// cout<<"The max_id memory allocated: "<<max_id_space/((double)1024*1024*1024)<<" GB ("<<((N_elements_max_id*100)/((double)vert_count*original_N_src_group))<< " %)"<<endl;
// cout<<"The last N_src_group: "<<N_src_group<<endl;
// // cout<<"The FQ/Next FQ: "<<(2*real_allocation*4)/((double)1024*1024*1024)<<" GB"<<endl; 
// // cout<<"The source_track_array/Next Next_source_track_array: "<<(2*real_allocation*4)/((double)1024*1024*1024)<<" GB"<<endl; 
// cout<<"Number of fill-ins detected: "<<Total_fill_count<<endl;
// cout<<"time for fill-in detection: "<<time<<endl;
// cout<<"N_dumping_cpu_memory: "<<Total_dumpings<<endl;

// // cout<<"Only dumping and loading time: "<<time_only_Copy*1000<<" ms"<<endl;
// cout<<"Achieved throughput of dumping and loading: "<< size_copied*sizeof(int)/((double)(1024*1024*1024))/(double)time_only_Copy<<" GB/s"<<endl;
// cout<<"N_elements_out-of-core (4 datastructures): "<<Total_Memory_Allocation - maxId_fill_size_threshhold<<endl;
// cout<<" out-of-core  datastructures size: "<<(Total_Memory_Allocation - maxId_fill_size_threshhold)*sizeof(uint_t)/((double)1024*1024*1024) <<" GB"<<endl;
// cout<<"N_src_group: "<<N_src_group<<endl;
// cout<<"4 datastructures space: "<<total_memory_allocation_in_GB/4<<" GB"<<endl;
// cout<<"CPU dumping and loading time: "<<dumping_loading_time*1000<<" ms"<<endl;
// cout<<"GPU:"<<GPU_slowest_time<<" Slowest virtual GPU takes: "<<maximum_time<< " ms"<<endl;
// cout<<"GPU:"<<GPU_fastest_time<<" Fastest virtual GPU takes: "<<minimum_time<< " ms"<<endl;
// cout<<"minimul4_out_core_allocation: "<<minimul4_out_core_allocation<<endl;
// cout<<"GPU:"<<GPU_least_edgeChecks<<" gets smallest edgechecks of: "<<min_edge_checks<<endl;
// cout<<"GPU:"<<GPU_highest_edgeChecks<<" gets highest edgechecks of: "<<max_edge_checks<<endl;

cout<<"vertex count: "<<vert_count<<" |4-data Allocation size: "<<total_memory_allocation_in_GB/4<<" GB"<<"  |beginning_N_src_group: "<<beginning_C<<"  |ending_N_src_group: "<<N_src_group<<"  |N_virtual_GPUs: "<<N_virtualGPU<<endl;
cout<<"Maximum frontier size: "<<maximum_front_size<<endl;
cout<<"Total Number of fill-ins detected: "<<Total_fill_count<<endl;
cout<<"Total_N_dumping_cpu_memory: "<<Total_dumpings<<endl;
// space_log_file<<"time for fill-in detection: "<<time<<endl;
cout<<"\tGPU:"<<GPU_slowest_time<<" Slowest virtual GPU takes: "<<maximum_time<< " ms"<<endl;
cout<<"\tGPU:"<<GPU_fastest_time<<" Fastest virtual GPU takes: "<<minimum_time<< " ms"<<endl;


// cout<<"\tSlowest GPU: CPU dumping and loading time: "<<TransferTime[GPU_slowest_time]*1000<<" ms"<<endl;
// // cout<<"Only dumping and loading time: "<<time_only_Copy*1000<<" ms"<<endl;
// cout<<"\tSlowest GPU: Achieved throughput of dumping and loading: "<< SIZE_COPIED[GPU_slowest_time]*sizeof(int)/((double)(1024*1024*1024))/(double)time_only_Copy<<" GB/s"<<endl;
// cout<<"\tSlowest GPU: N_dumping_cpu_memory: "<<All_N_dumping_CPU_memory[GPU_slowest_time]<<endl;

// // space_log_file<<"\tN_elements_out-of-core (4 datastructures): "<<Total_Memory_Allocation - max_id_size_threshhold<<endl;

// cout<<"\tMaximum All 4-out-of-core  datastructures size: "<<maximum_out_core_datasize*sizeof(uint_t)/((double)1024*1024*1024) <<" GB"<<endl;
// cout<<"\tMinimum All 4-out-of-core  datastructures size: "<<minimul4_out_core_allocation*sizeof(uint_t)/((double)1024*1024*1024) <<" GB"<<endl;
// // space_log_file<<"\tMaximum maxId[]  datastructures size: "<<(max_id_size_threshhold)*sizeof(uint_t)/((double)1024*1024*1024) <<" GB"<<endl;
// cout<<"\tfill[]  datastructures size: "<<(N_src_group*vert_count)*sizeof(uint_t)/((double)1024*1024*1024) <<" GB"<<endl;
cout<<"\tAll 6 datastructures size: "<<(Total_Memory_Allocation)*sizeof(uint_t)/((double)1024*1024*1024) <<" GB"<<endl;
cout<<"max_dumping_gpu: "<<max_dumping_gpu<<endl;
cout<<"#max_dumping: "<<max_dumping<<endl;
cout<<"max_dumping GPU transfer time: "<<TransferTime[max_dumping_gpu]*1000<<" ms"<<endl;

std::fstream space_log_file;
space_log_file.open("space_optimization_log_1_264_GPU_warp_centric.dat",std::fstream::out | std::fstream::app);
// space_log_file.open("test.dat",std::fstream::out | std::fstream::app);
space_log_file<<"vertex count: "<<vert_count<<" |Total_allocation: "<<total_memory_allocation_in_GB/4<<" GB"<<"  |beginning_N_src_group: "<<beginning_C<<"  |ending_N_src_group: "<<N_src_group<<"  |N_virtual_GPUs: "<<N_virtualGPU<<endl;
space_log_file<<"Total Number of fill-ins detected: "<<Total_fill_count<<endl;
space_log_file<<"Total_N_dumping_cpu_memory: "<<Total_dumpings<<endl;
// space_log_file<<"time for fill-in detection: "<<time<<endl;
space_log_file<<"\tGPU:"<<GPU_slowest_time<<" Slowest virtual GPU takes: "<<maximum_time<< " ms"<<endl;
space_log_file<<"\tGPU:"<<GPU_fastest_time<<" Fastest virtual GPU takes: "<<minimum_time<< " ms"<<endl;


// space_log_file<<"\tSlowest GPU: CPU dumping and loading time: "<<TransferTime[GPU_slowest_time]*1000<<" ms"<<endl;
// // cout<<"Only dumping and loading time: "<<time_only_Copy*1000<<" ms"<<endl;
// space_log_file<<"\tSlowest GPU: Achieved throughput of dumping and loading: "<< SIZE_COPIED[GPU_slowest_time]*sizeof(int)/((double)(1024*1024*1024))/(double)time_only_Copy<<" GB/s"<<endl;
// space_log_file<<"\tSlowest GPU: N_dumping_cpu_memory: "<<All_N_dumping_CPU_memory[GPU_slowest_time]<<endl;


// // space_log_file<<"\tN_elements_out-of-core (4 datastructures): "<<Total_Memory_Allocation - max_id_size_threshhold<<endl;

// space_log_file<<"\tMaximum All 4-out-of-core  datastructures size: "<<maximum_out_core_datasize*sizeof(uint_t)/((double)1024*1024*1024) <<" GB"<<endl;
// space_log_file<<"\tMinimum All 4-out-of-core  datastructures size: "<<minimum_out_core_datasize*sizeof(uint_t)/((double)1024*1024*1024) <<" GB"<<endl;
// // space_log_file<<"\tMaximum maxId[]  datastructures size: "<<(max_id_size_threshhold)*sizeof(uint_t)/((double)1024*1024*1024) <<" GB"<<endl;
// space_log_file<<"\tfill[]  datastructures size: "<<(N_src_group*vert_count)*sizeof(uint_t)/((double)1024*1024*1024) <<" GB"<<endl;
space_log_file<<"\tAll 6 datastructures size: "<<(Total_Memory_Allocation)*sizeof(uint_t)/((double)1024*1024*1024) <<" GB"<<endl;
space_log_file<<"max_dumping_gpu: "<<max_dumping_gpu<<endl;
space_log_file<<"#max_dumping: "<<max_dumping<<endl;
space_log_file<<"max_dumping GPU transfer time: "<<TransferTime[max_dumping_gpu]*1000<<" ms"<<endl;
// space_log_file<<"N_src_group: "<<N_src_group<<endl;
// space_log_file<<"4 datastructures space percentage: "<<total_memory_allocation_factor*100<<" %"<<endl;

space_log_file<<endl<<endl<<endl;
space_log_file.close();

return;
}
