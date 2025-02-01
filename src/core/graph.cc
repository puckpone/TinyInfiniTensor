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
    bool finished = false;
    while (!finished)
    {
        finished = true; // 假设本轮没有发生任何优化

        // 遍历所有算子 (以 prev 这个变量承接)
        for (auto &&prev : ops)
        {
            // 若当前算子不是 Transpose，则略过不处理
            if (prev->type != OpType::Transpose)
                continue;

            // 如果是 Transpose，则查看它的所有后继算子
            for (auto &&succ_w : prev->successors)
            {
                // succ_w 是一个 weak_ptr，需要 lock() 才能获得对应的 shared_ptr
                auto succ = succ_w.lock();
                if (!succ)
                    continue;

                // ============= 第一种情况：后继也是 Transpose =============
                // 若两个相邻 Transpose 做的是相同的 perm，则它们等效于“抵消”操作，可一起删除
                if (succ->type == OpType::Transpose)
                {
                    // 转成实际的 TransposeObj
                    auto tp_prev = as<TransposeObj>(prev);
                    auto tp_succ = as<TransposeObj>(succ);

                    // 比较二者的 perm
                    if (tp_prev->getPermute() == tp_succ->getPermute())
                    {
                        // 说明这两个 Transpose 连用等效于“恒等变换”，可删掉
                        finished = false; // 本轮发生了删除，需再继续循环

                        // 对 succ 的所有后继节点 (ss) 进行处理
                        //   - 把这些后继节点的输入从 succ->outputs[0] 改成 prev->inputs[0]，
                        //     即“跳过”这两个 Transpose
                        for (auto &&ss_w : succ->successors)
                        {
                            auto ss = ss_w.lock();
                            if (!ss)
                                continue;

                            // 遍历后继节点 ss 的所有输入
                            for (auto &&ss_input : ss->inputs)
                            {
                                // 如果该输入本来是 succ->outputs[0] (第二个 Transpose 的输出)，
                                // 就改成第一个 Transpose 的输入(=真正的原始输入)
                                if (ss_input == succ->outputs[0])
                                {
                                    // 先断开原先张量与ss的连接
                                    ss_input->removeTarget(ss);

                                    // removeTensor(ss_input) 会从图中移除这个中间张量
                                    removeTensor(ss_input);

                                    // 然后把 ss_input 改成 prev->inputs[0]
                                    ss_input = prev->inputs[0];

                                    // 也要把这个新张量跟当前 ss 建立连接
                                    ss_input->removeTarget(prev);
                                    ss_input->addTarget(ss);
                                }
                            }

                            // 调整前驱后继关系：去除 succ 这个前驱
                            ss->removePredecessors(succ);

                            // 把 prev 的前驱也接到 ss 上
                            for (auto prev_old_prev_w : prev->predecessors)
                            {
                                auto prev_old_prev = prev_old_prev_w.lock();
                                if (!prev_old_prev)
                                    continue;

                                // 前驱节点不再连到 prev
                                ss->addPredecessors(prev_old_prev);
                                prev_old_prev->removeSuccessors(prev);
                                prev_old_prev->addSuccessors(ss);
                            }
                        }

                        // 把第一个 Transpose 的输出张量删掉
                        for (auto &&prev_output : prev->outputs)
                        {
                            removeTensor(prev_output);
                        }

                        // 最后，把这两个 Transpose 算子从图中彻底移除
                        removeOperator(prev);
                        removeOperator(succ);

                        // 通过 goto 来跳出最外层的 for 循环，从而回到 while 循环再次扫描
                        goto next_round;
                    }
                }
                // ============= 第二种情况：后继是 MatMul =============
                // 如果前驱是 Transpose 且只交换了最后两个维度，那么把它合并进 MatMul 的 transA/transB
                else if (succ->type == OpType::MatMul)
                {
                    // 转成 TransposeObj / MatmulObj
                    auto tp_prev = as<TransposeObj>(prev);
                    auto mm_succ = as<MatmulObj>(succ);

                    auto perm = tp_prev->getPermute();
                    auto rank = perm.size();

                    // 检查是否只交换了最后两个维度
                    bool valid = true;
                    // 先看前面的维度是否不变
                    for (size_t i = 0; i < rank - 2; i++)
                    {
                        if (perm[i] != static_cast<int>(i))
                        {
                            valid = false;
                            break;
                        }
                    }
                    // 再看最后两个维度是否是互换
                    valid = valid &&
                            (perm[rank - 2] == static_cast<int>(rank - 1)) &&
                            (perm[rank - 1] == static_cast<int>(rank - 2));

                    if (!valid)
                    {
                        std::cout << "GGGGGGG\n";
                        continue;
                    }

                    // 满足只交换最后两个维度，则可以把这个 transpose 与 matmul 融合
                    finished = false; // 发生了合并，需再继续循环

                    // 先把 MatMul 从 prev (这个 transpose) 的前驱关系上断开
                    mm_succ->removePredecessors(prev);

                    // 把 prev 的前驱直接连到 MatMul
                    for (auto &&prev_old_prev_w : tp_prev->predecessors)
                    {
                        auto prev_old_prev = prev_old_prev_w.lock();
                        if (!prev_old_prev)
                            continue;
                        prev_old_prev->removeSuccessors(prev);
                        prev_old_prev->addSuccessors(succ);
                        mm_succ->addPredecessors(prev_old_prev);
                    }

                    // 修正 MatMul 的输入 (succ->inputs)
                    for (auto &&succ_input : succ->inputs)
                    {
                        // 如果某个输入是 prev->outputs[0]，就将它替换为 prev->inputs[0]
                        if (succ_input == prev->outputs[0])
                        {
                            // 决定是 transA 还是 transB
                            if (succ_input == succ->inputs[0])
                                mm_succ->setTransA(!mm_succ->getTransA());
                            if (succ_input == succ->inputs[1])
                                mm_succ->setTransB(!mm_succ->getTransB());

                            // 修改输入张量引用
                            succ_input = prev->inputs[0];
                            succ_input->removeTarget(prev);
                            succ_input->addTarget(succ);
                        }
                    }

                    // 把 Transpose 的输出张量删除
                    removeTensor(prev->outputs[0]);
                    // 删除该 Transpose 算子
                    removeOperator(prev);

                    // 跳出 for，回到 while(!finished) 再次扫描
                    goto next_round;
                }

            } // end for (auto &&succ_w : prev->successors)
        }     // end for (auto &&prev : ops)
    next_round:
        ;
        // 这里通过 goto + label 的方式进行多轮扫描
        // 如果上面没有任何优化发生，则 finished 保持 true，循环就会退出
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
    // topological sorting first
    IT_ASSERT(topo_sort() == true);

    // =================================== 作业 ===================================
    // TODO：利用 allocator 给计算图分配内存
    // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给tensor 绑定内存
    // =================================== 作业 ===================================
    auto n = this->tensors.size();
    vector<size_t> offsets(n);
    for (size_t i = 0; i < n; i++) {
        offsets[i] = this->allocator.alloc(this->tensors[i]->getBytes());
    }
    auto hptr = this->allocator.getPtr();
    for (size_t i = 0; i < n; i++) {
        auto ptr = static_cast<char*>(hptr) + offsets[i];
        auto blob = make_ref<BlobObj>(this->runtime, ptr);
        this->tensors[i]->setDataBlob(blob);
    }
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
