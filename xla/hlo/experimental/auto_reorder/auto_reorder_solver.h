#ifndef XLA_AUTO_REORDER_SOLVER_H_
#define XLA_AUTO_REORDER_SOLVER_H_
#include <limits>

#include <tuple>
#include <unordered_map>
#include <set>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/common_ortools_deps.h"


namespace xla {
  using IntVar = operations_research::sat::IntVar;
  using CpModelBuilder = operations_research::sat::CpModelBuilder;
  using IntervalVar = operations_research::sat::IntervalVar;
  namespace reorder{
    const uint32_t ksolveTimeout = 30;  // 30s
    
    static const int kChannelNumber = 2;
    bool solve_debug=false;
  }
  enum class NodeType { kCompute = 0, kCommunication = 1 };

struct TaskType {
  IntVar start;
  IntVar end;
  IntervalVar interval;
};
using CostType = int64_t;  // we can change it to double?

//TODO: using LPNode to abstract LPContainer and LPContainerDAG
template <typename ElementType>
class LPNode{
  public:
  virtual const std::string GetName() const = 0;
  virtual const int UUID() = 0;
  virtual CostType GetCost() const = 0;
  virtual void SetStart(CostType start) = 0;
  virtual CostType GetStart() = 0;
  virtual bool IsComputation() const = 0;
  virtual bool IsCommunication() const = 0;
  virtual NodeType GetType() const = 0;
  virtual bool HasValue() const = 0;
  virtual ElementType GetValue() const = 0;
  virtual void AddDep(LPNode* dep, CostType cost) = 0;
  virtual const std::vector<std::tuple<LPNode*, CostType>> GetDeps() const = 0;
  virtual void Freeze() = 0;
  private:
    std::vector<std::tuple<LPNode*, CostType>> deps_;
};

// LPContainer is a template class, it can be used to store any type of data
// 1. LPContainer<const HloInstruction*>; using to store one instruction
// 2. LPContainer<const LPContainerDAG>; using to store a graph of instructions,decrese lp hard
// 3. LPContainer<const Stage>; maybe we can use it to store a pipeline stage
template <typename ElementType>
class LPContainer{
  public:
  
  LPContainer(ElementType inner_element, CostType cost, NodeType type)
      : inner_element_(inner_element), cost_(cost), type_(type) {
    uuid_ = reinterpret_cast<uintptr_t>(this);
  };
  ~LPContainer() { deps_.clear(); };
  const std::string GetName() const { return inner_element_->ToShortString(); }
  const int UUID() { return inner_element_->unique_id(); }

  CostType GetCost() const { return cost_; }
  void SetStart(CostType start) { startat_ = start; }
  CostType GetStart() { return startat_; }
  bool IsComputation() const { return type_ == NodeType::kCompute; }
  bool IsCommunication() const { return type_ == NodeType::kCommunication; }
  NodeType GetType() const { return type_; }
  bool HasValue() const { return inner_element_ != nullptr; }
  ElementType GetValue() const { return inner_element_; }
  void AddDep(LPContainer* dep, CostType cost);
  const std::vector<std::tuple<LPContainer*, CostType>> GetDeps() const {
    return deps_;
  }
  void Freeze() { frozen_ = true; }

  private:
  CostType cost_;
  CostType startat_;
  NodeType type_;
  ElementType inner_element_;
  // deps store the edge
  std::vector<std::tuple<LPContainer*, CostType>> deps_;
  bool frozen_ = false;  // if it is frozen, it can not be changed,such as add deps
  uintptr_t uuid_;
  std::string name_;  // edge need a name

};
// LPContainerDAG is a graph of container, it can be used to store the DAG of container
// be used as a atomic unit of LPContainer
template <typename ElementType>
class LPContainerDAG{
  //we can use InstructionDAG to get memory effect order
  public:
    // maintain a DAG of inner elements
    struct DAGEdge{
      LPContainerDAG* from;
      LPContainerDAG* to;
      CostType cost;
    };
  //create a  LPContainerDAG with 
    LPContainerDAG(ElementType inner_element, CostType cost, NodeType type): cost_(cost), type_(type){
      inner_elements.push_back(LPContainer<ElementType>(inner_element, cost, type));
    };
    bool IsIn(LPContainerDAG<ElementType> *a){
      return users_.find(a) != users_.end();
    };
    //which container can be put together:1. they have the same type 2. they have dep between them
    static bool CanFused(LPContainerDAG<ElementType>* a, LPContainerDAG<ElementType>* b){

    };
    // AddChild
  private:
    
    std::set<ElementType> users_;
    std::set<ElementType> operands_;
    std::vector<LPContainer<ElementType>> inner_elements;
    CostType cost_;
  CostType startat_;
  NodeType type_;

};

// we only define node, edge is express by deps;
// edge is use to express the dependency between two nodes ，it have no effect
// constraint

//ContainerType is a template class, it can be used to store ElementType of data
//example: LPContainer<const HloInstruction*>; using to store one instruction, ElementType is const HloInstruction*, ContainerType is LPContainer<const HloInstruction*>
template <typename ContainerType,typename ElementType>
class LinearProgramScheduler {
  // https://developers.google.com/optimization/scheduling/job_shop?hl=zh-cn
  // be a linear programming problem or a integer programming problem,that's a
  // problem
 public:
  explicit LinearProgramScheduler(bool verbose = false) {
    cp_model_ = CpModelBuilder();
    verbose_ = verbose;
  };
  ~LinearProgramScheduler();
  // add Node to scheduler, so that Solve can use its deps
  Status AddConstraint(ContainerType* node);
  // solve the LP problem
  Status Solve();
  // find instruction,if not exist, return error
  StatusOr<ContainerType*> FindInstructionLPNode(ElementType instruction);
  // find LPNode by instruction,if not exist,create it
  ContainerType* FindLPNodeOrCreate(ElementType instruction, CostType cost,
                             NodeType type);
  // for debug: render graph viz
  void RenderGraphviz(std::string filename) const;
  // for debug: render gantt chart
  void RenderGantt(std::string filename) const;
  // set max start time as horizon
  void SetHorizon(uint32_t horizon) { horizon_ = horizon; }
  StatusOr<TaskType> FindTask(ContainerType* node);
  bool NodeHasAddTasks(ContainerType* node);
  CostType GetNodeStartTime(ContainerType* node);
  std::vector<ContainerType*> GetSortedNodes();
  void AddNodeToTask(ContainerType* node, TaskType task);
  StatusOr<TaskType> AddNodeToTask(ContainerType* node);
  
 private:
  CpModelBuilder cp_model_;
  bool verbose_ = false;
  std::unordered_map<int, ContainerType*> uuid2container;
  std::vector<ContainerType*> nodes_;
  uint32_t horizon_ = std::numeric_limits<uint32_t>::max();
  absl::flat_hash_map<int, std::tuple<ContainerType*, TaskType>>
      node_to_task_;  // every node hold interval_var,show what time it start
                      // and end
  // channels can be overlap each other
  std::map<NodeType, std::vector<IntervalVar>> channel_to_intervals_;
  std::map<int, int64_t> node_starttime_;
};
}  // namespace xla
#endif  // XLA_AUTO_REORDER_H_