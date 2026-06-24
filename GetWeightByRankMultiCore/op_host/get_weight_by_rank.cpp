#include "get_weight_by_rank_tiling.h"

#include <cstddef>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace GetWeightByRankConst;

namespace {
bool CheckVectorShape(const gert::Shape &shape, int64_t len)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == len;
}

bool CheckWeightShape(const gert::Shape &shape)
{
    return shape.GetDimNum() == 3 &&
           shape.GetDim(0) > 0 &&
           shape.GetDim(0) % INDEX_COUNT == 0 &&
           shape.GetDim(1) == RANKS_PER_INDEX &&
           shape.GetDim(2) == ROWS;
}

bool CheckSameWeightShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

int64_t NormalizeRankCount(int64_t rankCount)
{
    if (rankCount <= 0) {
        return 0;
    }
    return rankCount < RANKS_PER_INDEX ? rankCount : RANKS_PER_INDEX;
}

bool InferTotalOutputRowsImpl(const int64_t *lensData, const int64_t *rankData,
                              int64_t idxCount, int64_t totalUserEntries, int64_t &totalOutputRows)
{
    if (lensData == nullptr || rankData == nullptr || idxCount <= 0 || totalUserEntries < 0) {
        return false;
    }

    int64_t userOffset = 0;
    totalOutputRows = 0;
    for (int64_t i = 0; i < idxCount; ++i) {
        const int64_t currentLen = lensData[i] > 0 ? static_cast<int64_t>(lensData[i]) : 0;
        if (userOffset + currentLen > totalUserEntries) {
            return false;
        }

        int64_t groupRows = 0;
        for (int64_t k = 0; k < currentLen; ++k) {
            groupRows += NormalizeRankCount(rankData[userOffset + k]);
        }
        totalOutputRows += groupRows * INDEX_GROUP_WIDTH;
        userOffset += currentLen;
    }
    return userOffset == totalUserEntries && totalOutputRows > 0;
}

bool InferTotalOutputRows(const gert::Tensor *lensTensor, const gert::Tensor *ranksTensor,
                          int64_t idxCount, int64_t totalUserEntries, int64_t &totalOutputRows)
{
    if (lensTensor == nullptr || ranksTensor == nullptr ||
        lensTensor->GetDataType() != ranksTensor->GetDataType() ||
        lensTensor->GetShapeSize() != idxCount ||
        ranksTensor->GetShapeSize() != totalUserEntries) {
        return false;
    }
    if (lensTensor->GetDataType() == ge::DT_INT64) {
        return InferTotalOutputRowsImpl(lensTensor->GetData<int64_t>(), ranksTensor->GetData<int64_t>(),
                                        idxCount, totalUserEntries, totalOutputRows);
    }
    return false;
}

bool CheckOutputShape(const gert::Shape &shape)
{
    return shape.GetDimNum() == 2 &&
           shape.GetDim(0) > 0 &&
           shape.GetDim(1) == ROWS;
}

bool CheckSameOutputShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

UINT32 InferShapeFunc(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *weightRShape = context->GetInputShape(0);
    const gert::Shape *weightIShape = context->GetInputShape(1);
    const gert::Shape *idxsShape = context->GetInputShape(2);
    const gert::Shape *lensShape = context->GetInputShape(3);
    const gert::Shape *userIdsShape = context->GetInputShape(4);
    const gert::Shape *ranksShape = context->GetInputShape(5);
    gert::Shape *outRShape = context->GetOutputShape(0);
    gert::Shape *outIShape = context->GetOutputShape(1);
    if (weightRShape == nullptr || weightIShape == nullptr || idxsShape == nullptr ||
        lensShape == nullptr || userIdsShape == nullptr || ranksShape == nullptr ||
        outRShape == nullptr || outIShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckWeightShape(*weightRShape) || !CheckSameWeightShape(*weightRShape, *weightIShape) ||
        idxsShape->GetDimNum() != 1 || idxsShape->GetDim(0) <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t idxCount = idxsShape->GetDim(0);
    if (!CheckVectorShape(*lensShape, idxCount) ||
        userIdsShape->GetDimNum() != 1 ||
        ranksShape->GetDimNum() != 1 ||
        userIdsShape->GetDim(0) != ranksShape->GetDim(0)) {
        return ge::GRAPH_FAILED;
    }

    int64_t totalOutputRows = 0;
    if (!InferTotalOutputRows(context->GetInputTensor(3), context->GetInputTensor(5),
                              idxCount, userIdsShape->GetDim(0), totalOutputRows)) {
        return ge::GRAPH_FAILED;
    }

    *outRShape = gert::Shape({totalOutputRows, ROWS});
    *outIShape = gert::Shape({totalOutputRows, ROWS});
    return ge::GRAPH_SUCCESS;
}

UINT32 InferDataTypeFunc(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType weightRType = context->GetInputDataType(0);
    const ge::DataType weightIType = context->GetInputDataType(1);
    if (weightRType != ge::DT_FLOAT || weightIType != weightRType) {
        return ge::GRAPH_FAILED;
    }
    if (context->SetOutputDataType(0, weightRType) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(1, weightRType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto weightRShape = context->GetInputTensor(0)->GetOriginShape();
    auto weightIShape = context->GetInputTensor(1)->GetOriginShape();
    auto idxsShape = context->GetInputTensor(2)->GetOriginShape();
    auto lensShape = context->GetInputTensor(3)->GetOriginShape();
    auto userIdsShape = context->GetInputTensor(4)->GetOriginShape();
    auto ranksShape = context->GetInputTensor(5)->GetOriginShape();
    auto outRStorageShape = context->GetOutputShape(0);
    auto outIStorageShape = context->GetOutputShape(1);
    if (outRStorageShape == nullptr || outIStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto outRShape = outRStorageShape->GetOriginShape();
    auto outIShape = outIStorageShape->GetOriginShape();

    if (!CheckWeightShape(weightRShape) || !CheckSameWeightShape(weightRShape, weightIShape)) {
        return ge::GRAPH_FAILED;
    }
    if (idxsShape.GetDimNum() != 1 || idxsShape.GetDim(0) <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t userCount = weightRShape.GetDim(0) / INDEX_COUNT;
    const int64_t idxCount = idxsShape.GetDim(0);
    if (!CheckVectorShape(lensShape, idxCount) ||
        userIdsShape.GetDimNum() != 1 ||
        ranksShape.GetDimNum() != 1 ||
        userIdsShape.GetDim(0) != ranksShape.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t totalUserEntries = userIdsShape.GetDim(0);
    if (totalUserEntries < 0) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckOutputShape(outRShape) || !CheckSameOutputShape(outRShape, outIShape)) {
        return ge::GRAPH_FAILED;
    }
    int64_t inferredOutputRows = 0;
    if (InferTotalOutputRows(context->GetInputTensor(3), context->GetInputTensor(5),
                             idxCount, totalUserEntries, inferredOutputRows) &&
        outRShape.GetDim(0) != inferredOutputRows) {
        return ge::GRAPH_FAILED;
    }
    const int64_t totalOutputRows = outRShape.GetDim(0);

    GetWeightByRankTilingData tiling;
    tiling.set_userCount(static_cast<uint32_t>(userCount));
    tiling.set_idxCount(static_cast<uint32_t>(idxCount));
    tiling.set_totalUserEntries(static_cast<uint32_t>(totalUserEntries));
    tiling.set_totalOutputRows(static_cast<uint32_t>(totalOutputRows));

    const uint64_t dstCount = static_cast<uint64_t>(idxCount) * INDEX_GROUP_WIDTH;
    uint32_t blockDim = DEFAULT_BLOCK_DIM;
    if (dstCount > 0 && dstCount < DEFAULT_BLOCK_DIM) {
        blockDim = static_cast<uint32_t>(dstCount);
    }
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class GetWeightByRank : public OpDef {
public:
    explicit GetWeightByRank(const char *name) : OpDef(name)
    {
        this->Input("weight_r")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("weight_i")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("getIdxs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Input("lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .ValueDepend(REQUIRED);
        this->Input("getuserIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Input("getuserIdRank")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .ValueDepend(REQUIRED);
        this->Output("weightout_r")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("weightout_i")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(InferShapeFunc)
            .SetInferDataType(InferDataTypeFunc);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(GetWeightByRank);
} // namespace ops
