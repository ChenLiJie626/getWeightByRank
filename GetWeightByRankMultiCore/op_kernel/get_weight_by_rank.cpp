#include "kernel_operator.h"

using namespace AscendC;

namespace GetWeightByRankKernel {
constexpr uint32_t INDEX_COUNT = 136;
constexpr uint32_t ROWS = 384;
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

__aicore__ inline void ClearOneOutput(GlobalTensor<float> &dst, TBuf<> &zeroBuf,
                                      uint64_t totalElems, uint64_t blockIdx, uint64_t blockNum)
{
    LocalTensor<float> zeroLocal = zeroBuf.Get<float>();
    const uint64_t blockStep = blockNum * ZERO_CHUNK_ELEMS;
    for (uint64_t offset = blockIdx * ZERO_CHUNK_ELEMS; offset < totalElems; offset += blockStep) {
        const uint32_t count = static_cast<uint32_t>(MinU64(ZERO_CHUNK_ELEMS, totalElems - offset));
        Duplicate(zeroLocal, 0.0f, count);
        PipeBarrier<PIPE_ALL>();
        DataCopy(dst[offset], zeroLocal, count);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void ClearOutputs(GlobalTensor<float> &outR, GlobalTensor<float> &outI,
                                    TBuf<> &zeroBuf, uint64_t totalElems,
                                    uint64_t blockIdx, uint64_t blockNum)
{
    ClearOneOutput(outR, zeroBuf, totalElems, blockIdx, blockNum);
    ClearOneOutput(outI, zeroBuf, totalElems, blockIdx, blockNum);
}

__aicore__ inline void CopyRankColumnScalar(__gm__ float *dst, __gm__ float *src,
                                            uint64_t dstOffset, uint64_t srcOffset)
{
    for (uint32_t row = 0; row < ROWS; ++row) {
        dst[dstOffset + static_cast<uint64_t>(row) * OUTPUT_COLS] =
            src[srcOffset + static_cast<uint64_t>(row) * RANKS_PER_INDEX];
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

__aicore__ inline void ProcessCopies(GM_ADDR weightR, GM_ADDR weightI,
                                     GM_ADDR getIdxs, GM_ADDR lens,
                                     GM_ADDR getUserIds, GM_ADDR getUserIdRank,
                                     GM_ADDR weightoutR, GM_ADDR weightoutI,
                                     uint32_t userCount, uint32_t idxCount,
                                     uint32_t totalUserEntries)
{
    auto weightRPtr = reinterpret_cast<__gm__ float *>(weightR);
    auto weightIPtr = reinterpret_cast<__gm__ float *>(weightI);
    auto outRPtr = reinterpret_cast<__gm__ float *>(weightoutR);
    auto outIPtr = reinterpret_cast<__gm__ float *>(weightoutI);
    auto idxsGm = reinterpret_cast<__gm__ int32_t *>(getIdxs);
    auto lensGm = reinterpret_cast<__gm__ int32_t *>(lens);
    auto userIdsGm = reinterpret_cast<__gm__ int32_t *>(getUserIds);
    auto ranksGm = reinterpret_cast<__gm__ int32_t *>(getUserIdRank);

    uint32_t userOffset = 0;
    uint32_t outputColOffset = 0;
    for (uint32_t i = 0; i < idxCount; ++i) {
        const int32_t rawLen = lensGm[i];
        const uint32_t currentLen = rawLen > 0 ? static_cast<uint32_t>(rawLen) : 0;
        const uint32_t remaining = userOffset < totalUserEntries ? totalUserEntries - userOffset : 0;
        const uint32_t availableLen = currentLen < remaining ? currentLen : remaining;

        const int32_t idxGroup = idxsGm[i];
        const bool validIdxGroup = idxGroup >= 0 && static_cast<uint32_t>(idxGroup) < INDEX_GROUP_COUNT;
        for (uint32_t k = 0; k < availableLen; ++k) {
            const int32_t userId = userIdsGm[userOffset + k];
            const uint32_t rankCount = GetRankCount(ranksGm[userOffset + k]);
            const bool validUserId = IsValidUserId(userId, userCount);
            const uint32_t remainingCols =
                outputColOffset < OUTPUT_COLS ? OUTPUT_COLS - outputColOffset : 0;
            const uint32_t writeRankCount = rankCount < remainingCols ? rankCount : remainingCols;

            for (uint32_t rankOffset = 0; rankOffset < RANKS_PER_INDEX; ++rankOffset) {
                if (rankOffset >= writeRankCount) {
                    continue;
                }
                const uint32_t outputCol = outputColOffset + rankOffset;
                for (uint32_t offset = 0; offset < INDEX_GROUP_WIDTH; ++offset) {
                    if (!validIdxGroup || !validUserId) {
                        continue;
                    }

                    const uint32_t srcIndex = static_cast<uint32_t>(idxGroup) * INDEX_GROUP_WIDTH + offset;
                    const uint32_t dstIndex = i * INDEX_GROUP_WIDTH + offset;
                    const uint64_t srcOffset =
                        (((static_cast<uint64_t>(userId) * INDEX_COUNT + srcIndex) * ROWS) * RANKS_PER_INDEX) +
                        rankOffset;
                    const uint64_t dstOffset =
                        (static_cast<uint64_t>(dstIndex) * ROWS * OUTPUT_COLS) + outputCol;

                    CopyRankColumnScalar(outRPtr, weightRPtr, dstOffset, srcOffset);
                    CopyRankColumnScalar(outIPtr, weightIPtr, dstOffset, srcOffset);
                }
            }
            outputColOffset += rankCount;
        }

        if (currentLen >= remaining) {
            userOffset = totalUserEntries;
        } else {
            userOffset += currentLen;
        }
    }
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
    if (userCount == 0 || idxCount == 0) {
        return;
    }

    GlobalTensor<float> outRGm;
    GlobalTensor<float> outIGm;
    const uint64_t outElems = static_cast<uint64_t>(idxCount) * INDEX_GROUP_WIDTH * ROWS * OUTPUT_COLS;
    outRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightout_r), outElems);
    outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightout_i), outElems);

    TPipe pipe;
    TBuf<> zeroBuf;
    pipe.InitBuffer(zeroBuf, ZERO_CHUNK_ELEMS * FLOAT_BYTES);

    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t blockNum = GetAivBlockNum();
    ClearOutputs(outRGm, outIGm, zeroBuf, outElems, blockIdx, blockNum);

    if (blockIdx == 0) {
        ProcessCopies(weight_r, weight_i, getIdxs, lens, getuserIds, getuserIdRank,
                      weightout_r, weightout_i, userCount, idxCount, totalUserEntries);
    }
#endif
}
