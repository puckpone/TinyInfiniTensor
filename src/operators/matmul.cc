#include "operators/matmul.h"
#include "utils/operator_utils.h"
namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        
        //高维Tensor乘法就是最后两维进行矩阵乘法，前面的维度进行双向广播
        auto A = inputs[0], B = inputs[1];
        auto shapeA = A->getDims();
        auto shapeB = B->getDims();
        int rankA = A->getRank();
        int rankB = B->getRank();

        // 处理转置操作
        if (transA) {
            std::swap(shapeA[rankA - 1], shapeA[rankA - 2]);
        }
        if (transB) {
            std::swap(shapeB[rankB - 1], shapeB[rankB - 2]);
        }

        // 获取 A 和 B 的前 rank-2 维度
        Shape shapeA1(shapeA.begin(), shapeA.begin() + (rankA - 2));
        Shape shapeB1(shapeB.begin(), shapeB.begin() + (rankB - 2));

        // 推断广播后的形状
        Shape ret = infer_broadcast(shapeA1, shapeB1);
        if (ret.empty()) {
            ret = Shape{1};
        }

        // 确保矩阵乘法的内积维度匹配
        auto kA = shapeA[rankA - 1];
        auto kB = shapeB[rankB - 2];
        IT_ASSERT(kA == kB);

        // 获取输出矩阵的行数 m 和列数 n
        auto m = shapeA[rankA - 2];
        auto n = shapeB[rankB - 1];

        // 将行数 m 和列数 n 添加到广播后的形状中
        ret.emplace_back(m);
        ret.emplace_back(n);

        return {{ret}};
    }

} // namespace infini