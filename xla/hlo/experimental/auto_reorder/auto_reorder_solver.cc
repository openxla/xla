#include "xla/hlo/experimental/auto_reorder/auto_reorder_solver.h"
#include <fstream>
#include <iostream>

#ifndef LPSchedulerFunc(return_type)
#define LPSchedulerFunc(return_type) template <typename ContainerType,typename ElementType> return_type LinearProgramScheduler<ContainerType,ElementType>
#endif
namespace xla {
using IntVar = operations_research::sat::IntVar;
using CpModelBuilder = operations_research::sat::CpModelBuilder;
using IntervalVar = operations_research::sat::IntervalVar;
// namespace ORTools = operations_research::sat;
using Task =
    std::tuple<int8_t, CostType>;  // (channel, processing_time), we have two
                                   // channel now:communication and computation
using Job = std::vector<Task>;



template <typename ContainerType,typename ElementType>
LinearProgramScheduler<ContainerType, ElementType>::~LinearProgramScheduler() {
  uuid2container.clear();
  node_to_task_.clear();
  channel_to_intervals_.clear();
  // destroy nodes
  for (auto node : nodes_) {
    delete node;
  }
  nodes_.clear();
};
template<class T>
void LPContainer<T>::AddDep(LPContainer<T>* dep, CostType cost) {
  if (frozen_) {
    LOG(FATAL) << "Can not add dep to a frozen node";
    // raise exception
    return;
  }
  // every node should start after dep+cost
  deps_.push_back(std::make_tuple(dep, cost));
};

LPSchedulerFunc(StatusOr<ContainerType*>)::FindInstructionLPNode(
    ElementType instruction) {
  auto it = uuid2container.find(instruction->unique_id());

  if (it != uuid2container.end()) {
    return it->second;
  }
  TF_RET_CHECK(false) << "Can not find the node:" << instruction->ToString();
}
LPSchedulerFunc(ContainerType*)::FindLPNodeOrCreate(
    ElementType element, CostType cost, NodeType type) {
  auto it = uuid2container.find(element->unique_id());
  if (it != uuid2container.end()) {
    return it->second;
  }
  auto node = new ContainerType(element, cost, type);
  nodes_.push_back(node);
  uuid2container.emplace(element->unique_id(), node);
  return node;
};
LPSchedulerFunc(bool)::NodeHasAddTasks(ContainerType* node) {
  auto it = node_to_task_.find(node->UUID());
  return it != node_to_task_.end();
};
LPSchedulerFunc(void)::AddNodeToTask(ContainerType* node, TaskType task) {
}

LPSchedulerFunc(StatusOr<TaskType>)::FindTask(ContainerType* node) {
  auto it = node_to_task_.find(node->UUID());
  if (it != node_to_task_.end()) {
    return std::get<1>(it->second);
  } else {
    TF_RET_CHECK(false)
        << "Can not find the task for node:";  // << node->GetName();
  }
};
LPSchedulerFunc(Status)::AddConstraint(ContainerType* node) {
  if (NodeHasAddTasks(node)) {
    return OkStatus();
  }
  node->Freeze();
  return OkStatus();
};
LPSchedulerFunc(StatusOr<TaskType>)::AddNodeToTask(ContainerType* node) {
  IntVar start = cp_model_.NewIntVar({0, horizon_});
  IntVar end = cp_model_.NewIntVar({0, horizon_});
  IntervalVar interval = cp_model_.NewIntervalVar(start, node->GetCost(), end);
  TaskType task{start, end, interval};
  // AddNodeToTask(node, task);
  node_to_task_.emplace(node->UUID(), std::make_tuple(node, task));
  return task;
};
LPSchedulerFunc(tsl::Status)::Solve() {
  uint32_t max_execution_time = 0;
  for (auto node : nodes_) {
    max_execution_time += node->GetCost();
    for (auto dep_pair : node->GetDeps()) {
      auto cost = std::get<1>(dep_pair);
      max_execution_time += cost;
    }
  }
  SetHorizon(max_execution_time * reorder::kChannelNumber);

  for (auto node : nodes_) {
    VLOG(3) << "Add to scheduler" << node->GetName();
    TF_ASSIGN_OR_RETURN(TaskType node_task, AddNodeToTask(node));

    channel_to_intervals_[node->GetType()].push_back(node_task.interval);
    for (auto dep_pair : node->GetDeps()) {
      auto dep_node = std::get<0>(dep_pair);
      auto cost = std::get<1>(dep_pair);
      TaskType dep_task;
      TF_ASSIGN_OR_RETURN(dep_task, FindTask(dep_node));
      VLOG(3) << node->GetName() << "should start after" << dep_node->GetName()
              << "+" << cost;

      cp_model_.AddGreaterOrEqual(node_task.start, dep_task.end + cost);
    }
  }
  // add constraint, channels can be overlap each other
  for (auto it = channel_to_intervals_.begin();
       it != channel_to_intervals_.end(); it++) {
    cp_model_.AddNoOverlap(it->second);
  }
  //  objective.
  IntVar obj_var = cp_model_.NewIntVar({0, horizon_}).WithName("makespan");
  std::vector<IntVar> ends;
  for (auto it = node_to_task_.begin(); it != node_to_task_.end(); it++) {
    ends.push_back(std::get<1>(it->second).end);
  }
  cp_model_.AddMaxEquality(obj_var, ends);
  cp_model_.Minimize(obj_var);
  // cp_model_.
  // VLOG(2)<<"Number of variables:"<<cp_model_.NumVariables()<<" Number of
  // constraint:"<<cp_model_.NumConstraints();
  VLOG(1) << "Solving:" << node_to_task_.size() << " nodes";
  operations_research::sat::SatParameters parameters;
  parameters.set_max_time_in_seconds(reorder::ksolveTimeout);
  if(reorder::solve_debug){
    parameters.set_log_to_stdout(true);
    parameters.set_log_search_progress(true);
  }
  parameters.set_num_search_workers(8);
  const operations_research::sat::CpSolverResponse response =
      operations_research::sat::SolveWithParameters(cp_model_.Build(),
                                                    parameters);
  uint64_t solve_time = response.wall_time();
  VLOG(1) << "Solve finish:" << response.status()
          << " solve time:" << solve_time;

  if (response.status() == operations_research::sat::CpSolverStatus::OPTIMAL ||
      response.status() == operations_research::sat::CpSolverStatus::FEASIBLE) {
    VLOG(2) << "Optimal objective value:" << response.objective_value();
    for (auto kv : node_to_task_) {
      auto node_task_tuple = std::get<1>(kv);
      auto node = std::get<0>(node_task_tuple);
      auto task = std::get<1>(node_task_tuple);
      CostType start =
          operations_research::sat::SolutionIntegerValue(response, task.start);
      node->SetStart(start);
      VLOG(2) << node->GetName() << "should start at" << start << std::endl;
      node_starttime_.emplace(node->UUID(), start);
    }

    return OkStatus();
  } else {
    VLOG(2) << "Solve failed:" << response.status();
    return tsl::errors::NotFound("Linear Programming solve failed");
  }
};
std::string ReplaceUnusedChar(const std::string str,
                              const std::string need_move_str) {
  std::string result = str;
  for (auto c : need_move_str) {
    result.erase(std::remove(result.begin(), result.end(), c), result.end());
  }
  return result;
}
LPSchedulerFunc(void)::RenderGraphviz(std::string filename) const {
  // write a dot file
  std::string dot_file = absl::StrCat("/tmp/", filename, ".dot");
  std::ofstream out(dot_file);
  out << "digraph G {\n";
  VLOG(4) << "write node number:" << nodes_.size() << " to /tmp/" << filename
          << ".dot" << std::endl;
  auto get_node_name = [](const ContainerType* node) {
    return "\"" + ReplaceUnusedChar(node->GetName(), "%") + "\"";
  };
  bool draw_start_time = (node_starttime_.size() > 0);
  for (auto node : nodes_) {
    std::string color;
    if (node->IsCommunication()) {
      color = "orange";
    } else {
      color = "green";
    }
    if (draw_start_time) {
      out << get_node_name(node) << "[label=\""
          << ReplaceUnusedChar(node->GetName(), "") << "\\n"
          << "cost=" << node->GetCost()
          << "\nstart=" << node_starttime_.at(node->UUID())
          << "\",shape=box,color=" << color << "];\n";
    } else {
      out << get_node_name(node) << "[label=\""
          << ReplaceUnusedChar(node->GetName(), "") << "\\n"
          << "cost=" << node->GetCost() << "\",shape=box,color=" << color
          << "];\n";
    }

    for (auto dep_pair : node->GetDeps()) {
      auto dep_node = std::get<0>(dep_pair);
      auto dep_cost = std::get<1>(dep_pair);
      // draw edge
      out << get_node_name(dep_node) << "->" << get_node_name(node)
          << "[label=\"" << dep_cost << "\"];\n";
    }
  }
  out << "}\n";

  out.close();
  // convert dot file to png
  std::string png_file = absl::StrCat("/tmp/", filename, ".png");
  std::string cmd = absl::StrCat("dot -Tpng ", dot_file, " -o ", png_file);
  auto status = system(cmd.c_str());
  VLOG(4) << cmd << " execute status:" << status << std::endl;
}
LPSchedulerFunc(void)::RenderGantt(std::string filename) const {
  // https://g2.antv.antgroup.com/en/examples/storytelling/storytelling/#gantt
  // { name: 'compute',label:'kernel name1', startTime: 1, endTime: 4 },
  VLOG(4) << "write node number:" << nodes_.size() << " to /tmp/" << filename
          << ".js" << std::endl;
  auto get_node_name = [](const ContainerType* node) {
    return ReplaceUnusedChar(node->GetName(), "'");
  };
  bool draw_start_time = (node_starttime_.size() > 0);
  std::string csv_file = absl::StrCat("/tmp/", filename, ".js");
  std::ofstream csv_out(csv_file);
  csv_out << R"(import { Chart } from '@antv/g2'; 
  const events = [ )";
  for (auto node : nodes_) {
    std::string name;
    if (node->IsCommunication()) {
      name = "communication";
    } else {
      name = "compute";
    }
    if (draw_start_time) {
      csv_out << "{ name: \"" << name << "\",label:'"
              << ReplaceUnusedChar(node->GetName(), "'")
              << "', startTime: " << node_starttime_.at(node->UUID())
              << ", endTime: "
              << node_starttime_.at(node->UUID()) + node->GetCost() << " },\n";
    }
  }
  csv_out << "];";

  csv_out << R"(
  const chart = new Chart({
    container: 'container',
    autoFit: true,
  });

  chart.coordinate({ transform: [{ type: 'transpose' }] });

  chart
    .interval()
    .data(events)
    .encode('x', 'name')
    .encode('y', ['endTime', 'startTime'])
    .encode('color', 'name')
    .label({
      text: 'label',
      position: 'inside',
      transform: [{ type: 'overflowHide' }],
    })
    .encode('enterDuration', (d) => d.endTime - d.startTime)
    .encode('enterDelay', 'startTime')
    .scale('enterDuration', {
      zero: true,
      range: [0, 3000],
    });

  chart.render();)";
}
LPSchedulerFunc(std::vector<ContainerType*>)::GetSortedNodes() {
  std::vector<ContainerType*> sorted_nodes;
  sorted_nodes.reserve(nodes_.size());
  for (auto node : nodes_) {
    sorted_nodes.push_back(node);
  }
  std::sort(
      sorted_nodes.begin(), sorted_nodes.end(),
      [this](ContainerType* a, ContainerType* b) { return a->GetStart() < b->GetStart(); });
  return sorted_nodes;
}
template class LPContainer<const HloInstruction*>;
template class LinearProgramScheduler<LPContainer<const HloInstruction*>, const HloInstruction*>;
}  // namespace xla
