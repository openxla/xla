#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"

template <ck_tile::index_t M_Tile,
          ck_tile::index_t N_Tile,
          ck_tile::index_t K_Tile,
          ck_tile::index_t M_Warp,
          ck_tile::index_t N_Warp,
          ck_tile::index_t K_Warp,
          ck_tile::index_t M_Warp_Tile,
          ck_tile::index_t N_Warp_Tile,
          ck_tile::index_t K_Warp_Tile,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          bool kPadM = true,
          bool kPadN = true,
          bool kPadK = true,
          bool TransposeC = false,
          ck_tile::index_t TilePartitionerGroupNum = 8,
          ck_tile::index_t TilePartitionerM01 = 4,
          bool has_hot_loop = false,
          ck_tile::TailNumber tail_number = ck_tile::TailNumber::Full>
struct CkGemmKernelHelper {
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
        ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
        ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
        GemmShape, TilePartitionerGroupNum, TilePartitionerM01>;

    using TileGemmTraits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;
    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
        kPadM, kPadN, kPadK, ALayout, BLayout, CLayout, TransposeC>;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
        ADataType, BDataType, AccDataType, GemmShape, TileGemmTraits>;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
        ADataType, BDataType, AccDataType, GemmShape, GemmUniversalTraits, 
        ck_tile::GemmPipelineScheduler::Intrawave, has_hot_loop, tail_number>;

    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<
        UniversalGemmProblem, ck_tile::UniversalGemmPipelineAgBgCrPolicy>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<
            AccDataType, CDataType, CLayout,
            GemmPipelineProblem::kBlockSize,
            TilePartitioner::MPerBlock,
            TilePartitioner::NPerBlock,
            M_Warp, N_Warp,
            M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,
            UniversalGemmProblem::TransposeC>>;

    using GemmKernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
};

