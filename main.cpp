#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int myRank;	
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // get the number of devices
    int devices;
    cudaGetDeviceCount(&devices);

    // split MPI_COMM_WORLD into shared memory communicator
    MPI_Comm shardMemComm;
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, info, &shardMemComm);

    // get the id of the rank within the shared memory communicator
    // this effectively gives the id of the rank on its compute node
    int myRankOnNode;	
    MPI_Comm_rank(shardMemComm, &myRankOnNode);

    // set the device based on the id of the rank on the node
    // % number of devices to allow for oversubscription
    cudaSetDevice(myRankOnNode%devices);

    // printing of device selection
    int assignedDevice;
    cudaGetDevice(&assignedDevice);
    char hostname[128];
    int resLen;
    MPI_Get_processor_name(hostname, &resLen);
    std::cout << "rank " << myRank << " is rank " << myRankOnNode << " on host " << std::string(hostname) << " and was assigned device " << assignedDevice << std::endl;
}