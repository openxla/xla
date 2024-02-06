#ifndef XLA_AUTO_REORDER_H_
#define XLA_AUTO_REORDER_H_
#include "absl/strings/string_view.h"
#include "xla/hlo/experimental/auto_reorder/auto_reorder_solver.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/backend.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/model/analytical_latency_estimator.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/latency_hiding_scheduler.h"

// #include "xla/statusor.h"
namespace xla {
class AutoReorderPass : public HloModulePass {
 public:
  AutoReorderPass(){};
  AutoReorderPass(std::unique_ptr<LatencyEstimator> latency_estimator,
                  std::unique_ptr<AsyncTracker> async_tracker,
                  std::unique_ptr<SchedulerCore> scheduler_core,
                  HloCostAnalysis::ShapeSizeFunction shape_size_bytes)
      : async_tracker_(std::move(async_tracker)),
        scheduler_core_(std::move(scheduler_core)),
        latency_estimator_(std::move(latency_estimator)),
        shape_size_bytes_(shape_size_bytes){};
  absl::string_view name() const override { return "auto-reorder"; }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
  // when computation is changed, we need to rebuild the hlo ordering
  tsl::Status RebuildHloOrdering(HloSchedule& module_schedule,
                                 HloComputation* entry_computation);
  tsl::Status MoveInstruction(HloComputation* src_computation,
                              absl::string_view src_name,
                              HloComputation* dst_computation);
  int64_t OriginalInstructionPosition(const HloInstruction* instr) const {
    auto it = instr_order_map_.find(instr);
    CHECK(it != instr_order_map_.end());
    return it->second;
  }
  tsl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      HloComputation* computation);
  CostType GetInstructionStart(const HloInstruction* instr) const {
    auto it = instr_order_map_.find(instr);
    CHECK(it != instr_order_map_.end());
    return it->second;
  }
  void LogScheduleStatistics(const HloComputation* computation) {
    XLA_VLOG_LINES(1, LatencyHidingScheduler::SchedulerStatisticsString(
                          LatencyHidingScheduler::LatencyHidingStatistics(
                              computation, latency_estimator_.get(),
                              async_tracker_.get(), shape_size_bytes_)));
  }

 private:
  std::unique_ptr<AsyncTracker> async_tracker_;
  std::unique_ptr<SchedulerCore> scheduler_core_;
  std::unique_ptr<LatencyEstimator> latency_estimator_;
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloGraphNode>>
      nodes_;
  absl::flat_hash_map<const HloInstruction*, int64_t> instr_order_map_;
  // std::unique_ptr<LinearProgramScheduler> solver_;
  int64_t move_cost_threshold_in_bytes_;
  HloCostAnalysis::ShapeSizeFunction shape_size_bytes_;
};

CanonicalAsyncOp GpuGetCanonicalAsyncOp(const HloInstruction& hlo);

}  // namespace xla

#endif