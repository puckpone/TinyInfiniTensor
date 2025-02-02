#include "core/graph.h"
#include "core/op_type.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <queue>

namespace infini {

void GraphObj::addOperatorAndConnect(const Operator &op) {
    sorted = false;
    ops.push_back(op);
    for (auto &input : op->getInputs()) {
        if (input) {
            input->addTarget(op);
            if (auto pred = input->getSource()) {
                pred->addSuccessors(op);
                op->addPredecessors(pred);
            }
        }
    }
    for (auto &output : op->getOutputs()) {
        if (output) {
            output->setSource(op);
            for (auto &succ : output->getTargets()) {
                succ->addPredecessors(op);
                op->addSuccessors(succ);
            }
        }
    }
}

string GraphObj::toString() const {
    std::ostringstream oss;
    oss << "Graph Tensors:\n";
    for (const auto &tensor : tensors)
        oss << tensor << "\n";

    oss << "Graph operators:\n";
    for (const auto &op : ops) {
        vector<UidBaseType> preds, succs;
        for (auto &o : op->getPredecessors())
            preds.emplace_back(o->getGuid());
        for (auto &o : op->getSuccessors())
            succs.emplace_back(o->getGuid());
        oss << "OP " << op->getGuid();
        oss << ", pred " << vecToString(preds);
        oss << ", succ " << vecToString(succs);
        oss << ", " << op << "\n";
    }
    return oss.str();
}

bool GraphObj::topo_sort() {
    if (this->sorted) {
        return true;
    }
    std::vector<Operator> sorted;
    std::unordered_set<OperatorObj *> flags;
    sorted.reserve(ops.size());
    flags.reserve(ops.size());
    while (sorted.size() < ops.size()) {
        // Any node is move to sorted in this loop.
        auto modified = false;
        for (auto const &op : ops) {
            if (auto const &inputs = op->getInputs();
                flags.find(op.get()) == flags.end() &&
                std::all_of(inputs.begin(), inputs.end(),
                            [&flags](auto const &input) {
                                auto ptr = input->getSource().get();
                                return !ptr || flags.find(ptr) != flags.end();
                            })) {
                modified = true;
                sorted.emplace_back(op);
                flags.insert(op.get());
            }
        }
        if (!modified) {
            return false;
        }
    }
    this->ops = std::move(sorted);
    return this->sorted = true;
}

void GraphObj::optimize() {
    // =================================== 作业 ===================================
    // TODO: 设计一个算法来实现指定的图优化规则
    // 图优化规则如下：
    // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
    // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
    // =================================== 作业 ===================================
    // 使用一个布尔标志位finished，表示当前优化过程是否已完成
    // 若在循环中发现能继续做优化(删除算子/合并算子)，则设finished=false，并再次循环
    bool finished = false;   // 标记是否完成所有优化
    while (!finished) {
        finished = true;     // 默认假设本轮没有发生任何优化
        bool needRestart = false; // 标记本轮是否发生了优化，若发生则需要重新扫描

        // 遍历图中所有算子
        for (auto &&prev : ops) {

            // 若不是 Transpose，则跳过
            if (prev->type != OpType::Transpose)
                continue;

            // 遍历当前 Transpose 的所有后继算子
            for (auto &&succ_w : prev->successors) {
                auto succ = succ_w.lock();
                if (!succ)
                    continue;

                // 情况1：后继也是 Transpose，且两者 permute 相同时，相当于抵消，删除
                if (succ->type == OpType::Transpose) {
                    auto tp_prev = as<TransposeObj>(prev);
                    auto tp_succ = as<TransposeObj>(succ);

                    // 如果 permute 相同，则可删去这两个 Transpose
                    if (tp_prev->getPermute() == tp_succ->getPermute()) {
                        finished = false;
                        needRestart = true;

                        // 处理后继节点的后继（即将其输入指向 prev 的输入）
                        for (auto &&ss_w : succ->successors) {
                            auto ss = ss_w.lock();
                            if (!ss)
                                continue;
                            for (auto &&ss_input : ss->inputs) {
                                if (ss_input == succ->outputs[0]) {
                                    ss_input->removeTarget(ss);
                                    removeTensor(ss_input);
                                    ss_input = prev->inputs[0];
                                    ss_input->removeTarget(prev);
                                    ss_input->addTarget(ss);
                                }
                            }
                            ss->removePredecessors(succ);
                            // 把 prev 的前驱设为此节点的前驱
                            for (auto prev_old_prev_w : prev->predecessors) {
                                auto prev_old_prev = prev_old_prev_w.lock();
                                if (!prev_old_prev)
                                    continue;
                                ss->addPredecessors(prev_old_prev);
                                prev_old_prev->removeSuccessors(prev);
                                prev_old_prev->addSuccessors(ss);
                            }
                        }
                        // 删除 prev 的输出张量并删除两个 Transpose 算子
                        for (auto &&prev_output : prev->outputs) {
                            removeTensor(prev_output);
                        }
                        removeOperator(prev);
                        removeOperator(succ);
                        break; // 退出当前后继循环
                    }
                }
                // 情况2：后继是 MatMul，且该 Transpose 只交换最后两个维度
                else if (succ->type == OpType::MatMul) {
                    auto tp_prev = as<TransposeObj>(prev);
                    auto mm_succ = as<MatmulObj>(succ);

                    auto perm = tp_prev->getPermute();
                    auto rank = perm.size();
                    bool valid = true;

                    // 检查是否只交换了最后两个维度
                    for (size_t i = 0; i < rank - 2; i++) {
                        if (perm[i] != static_cast<int>(i)) {
                            valid = false;
                            break;
                        }
                    }
                    valid = valid &&
                            (perm[rank - 2] == static_cast<int>(rank - 1)) &&
                            (perm[rank - 1] == static_cast<int>(rank - 2));

                    if (!valid) {
                        // 删除原有的输出调试信息
                        continue;
                    }

                    // 合并到 MatMul 的 transA/transB
                    finished = false;
                    needRestart = true;
                    mm_succ->removePredecessors(prev);

                    // 替换前驱关系
                    for (auto &&prev_old_prev_w : tp_prev->predecessors) {
                        auto prev_old_prev = prev_old_prev_w.lock();
                        if (!prev_old_prev)
                            continue;
                        prev_old_prev->removeSuccessors(prev);
                        prev_old_prev->addSuccessors(succ);
                        mm_succ->addPredecessors(prev_old_prev);
                    }
                    // 更新 MatMul 的输入，并设置 transA/transB
                    for (auto &&succ_input : succ->inputs) {
                        if (succ_input == prev->outputs[0]) {
                            if (succ_input == succ->inputs[0])
                                mm_succ->setTransA(!mm_succ->getTransA());
                            if (succ_input == succ->inputs[1])
                                mm_succ->setTransB(!mm_succ->getTransB());

                            succ_input = prev->inputs[0];
                            succ_input->removeTarget(prev);
                            succ_input->addTarget(succ);
                        }
                    }
                    // 删除该 Transpose 的输出张量以及算子
                    removeTensor(prev->outputs[0]);
                    removeOperator(prev);
                    break; // 退出当前后继循环
                }
            }

            // 若本轮已进行优化操作，需要重新扫描，跳出当前for循环
            if (needRestart) {
                break;
            }
        }

        // 若本轮有优化发生，则继续重新扫描
        if (needRestart) {
            continue;
        }
    }
}

Tensor GraphObj::getTensor(int fuid) const {
    for (auto tensor : tensors) {
        if (tensor->getFuid() == fuid) {
            return tensor;
        }
    }
    return nullptr;
}

void GraphObj::shape_infer() {
    for (auto &op : ops) {
        auto ans = op->inferShape();
        IT_ASSERT(ans.has_value());
        auto oldOutputs = op->getOutputs();
        IT_ASSERT(ans.value().size() == oldOutputs.size());
        // replace the old outputshape and size with new one
        for (int i = 0; i < (int)ans.value().size(); ++i) {
            auto newShape = ans.value()[i];
            auto oldShape = oldOutputs[i]->getDims();
            auto fuid = oldOutputs[i]->getFuid();
            if (newShape != oldShape) {
                auto tensor = this->getTensor(fuid);
                tensor->setShape(newShape);
            }
        }
    }
}


void GraphObj::dataMalloc() {
    // 首先进行拓扑排序
    IT_ASSERT(topo_sort() == true);

    // 判断是否需要先为所有张量预分配总内存
    size_t totalSize = 0;
    for(auto &tensor : tensors){
        totalSize += tensor->getBytes();  // 计算总所需内存
    }
    auto basePtr = this->allocator.alloc(totalSize);  // 一次性分配总内存
    auto hptr = this->allocator.getPtr();  // 获取已分配内存的起始指针

    // 确保内存已成功分配
    IT_ASSERT(hptr != nullptr);

    // 为每个 tensor 分配内存
    size_t currentOffset = 0;
    for(auto &tensor : tensors){
        auto size = tensor->getBytes();
        auto blob = make_ref<BlobObj>(this->runtime, static_cast<char*>(hptr) + currentOffset);
        tensor->setDataBlob(blob);
        currentOffset += size;  // 更新偏移量
    }

    // 输出内存分配信息
    allocator.info();
}



Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
    return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
    IT_ASSERT(tensor->getRuntime() == runtime,
              std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                  tensor->getRuntime()->toString() + " to " +
                  runtime->toString());
    tensors.emplace_back(tensor);
    return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
    for (auto &t : tensors)
        addTensor(t);
    return tensors;
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
    for (auto tensor : tensors) {
        IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                    nullptr == tensor->getSource()));
        for (auto op : tensor->getTargets()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
        }
        auto op = tensor->getSource();
        IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
    }
    for (auto op : ops) {
        for (auto tensor : op->getInputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }
        for (auto tensor : op->getOutputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }
        for (auto pre : op->getPredecessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
        }
        for (auto suc : op->getSuccessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
        }
    }
    std::set<UidBaseType> s;
    // check whether two tensors with the same FUID exist
    for (auto tensor : tensors) {
        int cnt = s.count(tensor->getFuid());
        IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
        s.insert(tensor->getFuid());
    }
    return true;
}

} // namespace infini
