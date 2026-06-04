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
    return shape.GetDimNum() == 4 &&
           shape.GetDim(0) > 0 &&
           shape.GetDim(1) == INDEX_COUNT &&
           shape.GetDim(2) == ROWS &&
           shape.GetDim(3) == RANKS_PER_INDEX;
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

bool CheckOutputShape(const gert::Shape &shape, int64_t idxCount)
{
    return shape.GetDimNum() == 3 &&
           shape.GetDim(0) == idxCount * INDEX_GROUP_WIDTH &&
           shape.GetDim(1) == ROWS &&
           shape.GetDim(2) == OUTPUT_COLS;
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
    const int64_t userCount = weightRShape.GetDim(0);
    const int64_t idxCount = idxsShape.GetDim(0);
    if (!CheckVectorShape(lensShape, idxCount) ||
        userIdsShape.GetDimNum() != 1 ||
        ranksShape.GetDimNum() != 1 ||
        userIdsShape.GetDim(0) != ranksShape.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t totalUserEntries = userIdsShape.GetDim(0);
    if (totalUserEntries < 0 ||
        static_cast<uint64_t>(totalUserEntries) > static_cast<uint64_t>(idxCount) * OUTPUT_COLS) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckOutputShape(outRShape, idxCount) ||
        !CheckOutputShape(outIShape, idxCount)) {
        return ge::GRAPH_FAILED;
    }

    GetWeightByRankTilingData tiling;
    tiling.set_userCount(static_cast<uint32_t>(userCount));
    tiling.set_idxCount(static_cast<uint32_t>(idxCount));
    tiling.set_totalUserEntries(static_cast<uint32_t>(totalUserEntries));

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
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Input("getuserIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Input("getuserIdRank")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Output("weightout_r")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("weightout_i")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(GetWeightByRank);
} // namespace ops
