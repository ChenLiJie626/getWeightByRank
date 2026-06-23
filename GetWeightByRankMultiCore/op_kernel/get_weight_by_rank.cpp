#include "kernel_operator.h"

using namespace AscendC;

namespace GetWeightByRankKernel {
constexpr uint32_t INDEX_COUNT = 136;
constexpr uint32_t ROWS = 256;
constexpr uint32_t RANKS_PER_INDEX = 8;
constexpr uint32_t INDEX_GROUP_WIDTH = 8;
constexpr uint32_t INDEX_GROUP_COUNT = INDEX_COUNT / INDEX_GROUP_WIDTH;
constexpr uint32_t OUTPUT_COLS = 8;

constexpr uint32_t FLOAT_BYTES = sizeof(float);
constexpr uint32_t ZERO_CHUNK_ELEMS = 1024;

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t GetAivBlockNum()
{
    return static_cast<uint64_t>(GetBlockNum());
}

__aicore__ inline void ClearLocalMatrix(LocalTensor<float> &dst, uint32_t elemCount)
{
    for (uint32_t offset = 0; offset < elemCount; offset += ZERO_CHUNK_ELEMS) {
        const uint32_t count = static_cast<uint32_t>(MinU64(ZERO_CHUNK_ELEMS, elemCount - offset));
        Duplicate(dst[offset], 0.0f, count);
    }
}

__aicore__ inline bool IsValidUserId(int32_t userId, uint32_t userCount)
{
    return userId >= 0 && static_cast<uint32_t>(userId) < userCount;
}

__aicore__ inline uint32_t GetRankCount(int32_t rankCount)
{
    if (rankCount <= 0) {
        return 0;
    }
    const uint32_t count = static_cast<uint32_t>(rankCount);
    return count < RANKS_PER_INDEX ? count : RANKS_PER_INDEX;
}

__aicore__ inline uint32_t GetUserOffset(__gm__ int32_t *lensGm, uint32_t groupCount,
                                         uint32_t totalUserEntries)
{
    uint32_t userOffset = 0;
    for (uint32_t i = 0; i < groupCount; ++i) {
        const int32_t rawLen = lensGm[i];
        const uint32_t currentLen = rawLen > 0 ? static_cast<uint32_t>(rawLen) : 0;
        const uint32_t remaining = userOffset < totalUserEntries ? totalUserEntries - userOffset : 0;
        if (currentLen >= remaining) {
            return totalUserEntries;
        }
        userOffset += currentLen;
    }
    return userOffset;
}

__aicore__ inline void ProcessDstIndex(GlobalTensor<float> &weightR, GlobalTensor<float> &weightI,
                                       GM_ADDR getIdxs, GM_ADDR lens,
                                       GM_ADDR getUserIds, GM_ADDR getUserIdRank,
                                       GlobalTensor<float> &outR, GlobalTensor<float> &outI,
                                       TBuf<> &outRBuf, TBuf<> &outIBuf,
                                       uint32_t userCount, uint32_t totalUserEntries,
                                       uint32_t outputRows, uint32_t outMatrixElems,
                                       uint32_t dstIndex)
{
    auto idxsGm = reinterpret_cast<__gm__ int32_t *>(getIdxs);
    auto lensGm = reinterpret_cast<__gm__ int32_t *>(lens);
    auto userIdsGm = reinterpret_cast<__gm__ int32_t *>(getUserIds);
    auto ranksGm = reinterpret_cast<__gm__ int32_t *>(getUserIdRank);

    const uint32_t groupIndex = dstIndex / INDEX_GROUP_WIDTH;
    const uint32_t localIndex = dstIndex % INDEX_GROUP_WIDTH;
    const uint64_t baseDstOffset = static_cast<uint64_t>(dstIndex) * outMatrixElems;

    LocalTensor<float> outRLocal = outRBuf.Get<float>();
    LocalTensor<float> outILocal = outIBuf.Get<float>();
    ClearLocalMatrix(outRLocal, outMatrixElems);
    ClearLocalMatrix(outILocal, outMatrixElems);
    PipeBarrier<PIPE_ALL>();

    const uint32_t userOffset = GetUserOffset(lensGm, groupIndex, totalUserEntries);
    uint32_t outputColOffset = 0;

    const int32_t rawLen = lensGm[groupIndex];
    const uint32_t currentLen = rawLen > 0 ? static_cast<uint32_t>(rawLen) : 0;
    const uint32_t remaining = userOffset < totalUserEntries ? totalUserEntries - userOffset : 0;
    const uint32_t availableLen = currentLen < remaining ? currentLen : remaining;
    const int32_t idxGroup = idxsGm[groupIndex];
    if (idxGroup < 0 || static_cast<uint32_t>(idxGroup) >= INDEX_GROUP_COUNT) {
        DataCopy(outR[baseDstOffset], outRLocal, outMatrixElems);
        DataCopy(outI[baseDstOffset], outILocal, outMatrixElems);
        PipeBarrier<PIPE_ALL>();
        return;
    }
    const uint32_t srcIndex = static_cast<uint32_t>(idxGroup) * INDEX_GROUP_WIDTH + localIndex;

    for (uint32_t k = 0; k < availableLen; ++k) {
        const int32_t userId = userIdsGm[userOffset + k];
        const uint32_t rankCount = GetRankCount(ranksGm[userOffset + k]);
        const uint32_t remainingCols =
            outputColOffset < outputRows ? outputRows - outputColOffset : 0;
        const uint32_t copyCols = rankCount < remainingCols ? rankCount : remainingCols;
        if (copyCols > 0 && IsValidUserId(userId, userCount)) {
            const uint64_t baseSrcOffset =
                ((static_cast<uint64_t>(userId) * INDEX_COUNT + srcIndex) * RANKS_PER_INDEX) * ROWS;
            const uint32_t dstLocalOffset = outputColOffset * ROWS;
            const uint32_t copyElems = copyCols * ROWS;
            DataCopy(outRLocal[dstLocalOffset], weightR[baseSrcOffset], copyElems);
            DataCopy(outILocal[dstLocalOffset], weightI[baseSrcOffset], copyElems);
            PipeBarrier<PIPE_ALL>();
        }
        outputColOffset += rankCount;
    }

    DataCopy(outR[baseDstOffset], outRLocal, outMatrixElems);
    DataCopy(outI[baseDstOffset], outILocal, outMatrixElems);
    PipeBarrier<PIPE_ALL>();
}
} // namespace GetWeightByRankKernel

extern "C" __global__ __aicore__ void get_weight_by_rank(GM_ADDR weight_r, GM_ADDR weight_i,
                                                          GM_ADDR getIdxs, GM_ADDR lens,
                                                          GM_ADDR getuserIds, GM_ADDR getuserIdRank,
                                                          GM_ADDR weightout_r, GM_ADDR weightout_i,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(__DAV_CUBE__)
    (void)weight_r;
    (void)weight_i;
    (void)getIdxs;
    (void)lens;
    (void)getuserIds;
    (void)getuserIdRank;
    (void)weightout_r;
    (void)weightout_i;
    (void)workspace;
    (void)tiling;
    return;
#else
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    using namespace GetWeightByRankKernel;

    const uint32_t userCount = tilingData.userCount;
    const uint32_t idxCount = tilingData.idxCount;
    const uint32_t totalUserEntries = tilingData.totalUserEntries;
    const uint32_t outputRows = tilingData.outputRows;
    if (userCount == 0 || idxCount == 0) {
        return;
    }

    GlobalTensor<float> outRGm;
    GlobalTensor<float> outIGm;
    GlobalTensor<float> weightRGm;
    GlobalTensor<float> weightIGm;
    const uint32_t outMatrixElems = outputRows * ROWS;
    const uint32_t outMatrixBytes = outMatrixElems * FLOAT_BYTES;
    const uint64_t outElems = static_cast<uint64_t>(idxCount) * INDEX_GROUP_WIDTH * outMatrixElems;
    const uint64_t weightElems = static_cast<uint64_t>(userCount) * INDEX_COUNT * ROWS * RANKS_PER_INDEX;
    outRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightout_r), outElems);
    outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightout_i), outElems);
    weightRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight_r), weightElems);
    weightIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight_i), weightElems);

    TPipe pipe;
    TBuf<> outRBuf;
    TBuf<> outIBuf;
    pipe.InitBuffer(outRBuf, outMatrixBytes);
    pipe.InitBuffer(outIBuf, outMatrixBytes);

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(GetAivBlockNum());
    const uint32_t dstCount = idxCount * INDEX_GROUP_WIDTH;
    for (uint32_t dstIndex = blockIdx; dstIndex < dstCount; dstIndex += blockNum) {
        ProcessDstIndex(weightRGm, weightIGm, getIdxs, lens, getuserIds, getuserIdRank,
                        outRGm, outIGm, outRBuf, outIBuf,
                        userCount, totalUserEntries, outputRows, outMatrixElems, dstIndex);
    }
#endif
}
