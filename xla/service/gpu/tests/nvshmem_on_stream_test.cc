#include "nvshmem.h"
#include "nvshmemx.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "bootstrap_device_host/nvshmem_uniqueid.h"
#include "host/nvshmemx_coll_api.h"

void write_uid_to_file(const char *file_name, const nvshmemx_uniqueid_t &id) {
  std::ofstream file;
  file.open(file_name, std::ios::trunc | std::ios::out);
  file.write((char*)&id.version, sizeof(nvshmemx_uniqueid_v1));
  file.close();
}

void print_uuid(const nvshmemx_uniqueid_t &id) {
  for (int i = 0; i < sizeof(nvshmemx_uniqueid_v1); i++) {
    std::cout << (int)id.internal[i];
  }
  std::cout << std::endl;
}

nvshmemx_uniqueid_t read_uid_from_file(const char *file_name) {
  nvshmemx_uniqueid_t id;
  std::ifstream file(file_name, std::ios::in | std::ios::binary);
  file.read((char*)&id.version, sizeof(nvshmemx_uniqueid_v1));
  file.close();
  return id;
}

inline bool file_exists(const char *name) {
  std::ifstream f(name);
  return f.good();
}

#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (cudaSuccess != result) {                                               \
      fprintf(stdout, "[%s:%d] CUDA failed with %s \n", __FILE__, __LINE__,    \
              cudaGetErrorString(result));                                     \
      exit(-1);                                                                \
    }                                                                          \
  } while (false)

int main(int argc, char* argv[]) {
  std::cout << "[HOST LOGGING]: Argc count: " << argc << std::endl;
  if (argc != 3) {
    std::cout << "[HOST LOGGING]: Usage: " << argv[0]
        << " <rank> <exchange_file>" << std::endl;
    return 1;
  }

  int rank = std::stoi(argv[1]);
  std::string exchange_file = argv[2];
  std::cout << "[HOST LOGGING]: Rank: " << rank
            << ". Exchange file: " << exchange_file << std::endl;
  std::cout << "[HOST LOGGING]: Rank: " << rank << std::endl;
  int mype_node;
  int *input;
  int *output;
  int input_nelems = 8;

  nvshmemx_init_attr_t attr;
  nvshmemx_uniqueid_t id;
  if (rank == 0) {
    nvshmemx_get_uniqueid(&id);
    write_uid_to_file(exchange_file.c_str(), id);
    std::cout << "[HOST LOGGING]: Leader unique id: " << std::endl;
    print_uuid(id);
  } else {
    while (!file_exists(exchange_file.c_str())) {
      absl::SleepFor(absl::Seconds(1));
    }

    id = read_uid_from_file(exchange_file.c_str());
    print_uuid(id);
  }

  CUDA_CHECK(cudaSetDevice(rank));
  std::cout << "[HOST LOGGING]: Device set: " << rank << std::endl;
  std::cout << "[HOST LOGGING]: UID version: " << id.version << std::endl;
  nvshmemx_set_attr_uniqueid_args(/*rank=*/rank, /*num_ranks=*/2, &id, &attr);

  std::cout << "[HOST LOGGING]: nvshmemx_set_attr_uniqueid_args completed"
      << std::endl;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

  std::cout << "[HOST LOGGING]: NVSHMEM initialized." << std::endl;
  cudaStream_t compute_stream;
  CUDA_CHECK(cudaStreamCreate(&compute_stream));

  std::cout << "[HOST LOGGING]: Stream created." << std::endl;
  mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  std::cout << "[HOST LOGGING]: My PE node: " << mype_node << std::endl;

  input = (int *)nvshmem_malloc(sizeof(int) * input_nelems);
  std::cout << "[HOST LOGGING]: Input allocated." << std::endl;
  std::vector<int> input_host(input_nelems);

  for (int i = 0; i < input_nelems; i++) {
    input_host[i] = i + input_nelems * rank;
  }

  nvshmemx_putmem_on_stream(input, input_host.data(),
                            input_nelems * sizeof(int), mype_node,
                            compute_stream);

  CUDA_CHECK(cudaStreamSynchronize(compute_stream));
  nvshmemx_barrier_all_on_stream(compute_stream);

  
  
  {
    std::vector<int> input_back(input_nelems);
    CUDA_CHECK(cudaMemcpy(input_back.data(), input, input_back.size() * sizeof(int),
              cudaMemcpyDeviceToHost));

    std::cout << "[HOST LOGGING]: Input: ";
    for (int i = 0; i < input_nelems; i++) {
      std::cout << input_back[i] << " ";
    }
    std::cout << std::endl;
  }
  
  std::cout << "[HOST LOGGING]: Input copied to device." << std::endl;

  output = (int *)nvshmem_malloc(sizeof(int) * input_nelems);
  std::cout << "[HOST LOGGING]: Output memory allocated." << std::endl;
  
  nvshmemx_barrier_all_on_stream(compute_stream);
  CUDA_CHECK(cudaStreamSynchronize(compute_stream));
  nvshmem_sync_all();
  std::cout << "[HOST LOGGING]: Barrier done." << std::endl;
  
  nvshmemx_int_sum_reduce_on_stream(NVSHMEMX_TEAM_NODE, output, input, 1,
                                    compute_stream);
  std::cout << "[HOST LOGGING]: Reduce done." << std::endl;
  nvshmemx_barrier_all_on_stream(compute_stream);
  nvshmem_sync_all();

  std::cout << "[HOST LOGGING]: Stream synchronized." << std::endl;

  std::vector<int> input_back(input_nelems);
  CUDA_CHECK(cudaMemcpyAsync(input_back.data(), input, input_back.size() * sizeof(int),
            cudaMemcpyDeviceToHost, compute_stream));
  CUDA_CHECK(cudaStreamSynchronize(compute_stream));

  std::cout << "[HOST LOGGING]: Input: ";
  for (int i = 0; i < input_nelems; i++) {
    std::cout << input_back[i] << " ";
  }
  
  std::vector<int> output_host(input_nelems);
  CUDA_CHECK(cudaMemcpyAsync(output_host.data(), output, input_nelems * sizeof(int),
            cudaMemcpyDeviceToHost, compute_stream));
  CUDA_CHECK(cudaStreamSynchronize(compute_stream));

  std::cout << "[HOST LOGGING]: Output: ";
  for (int i = 0; i < input_nelems; i++) {
    std::cout << output_host[i] << " ";
  }
  std::cout << std::endl;

  CUDA_CHECK(cudaStreamSynchronize(compute_stream));
  CUDA_CHECK(cudaStreamDestroy(compute_stream));
  nvshmem_free(input);
  nvshmem_free(output);
  nvshmem_finalize();

  std::cout << "[HOST LOGGING]: Completed." << std::endl;
  return 0;
}