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
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t FLOATS_PER_DATA_BLOCK = DATA_BLOCK_BYTES / FLOAT_BYTES;
constexpr uint32_t PADDED_COLUMN_ELEMS = ROWS * FLOATS_PER_DATA_BLOCK;
constexpr uint32_t PADDED_COLUMN_BYTES = PADDED_COLUMN_ELEMS * FLOAT_BYTES;
constexpr uint32_t ZERO_CHUNK_ELEMS = 1024;
constexpr uint32_t VECTOR_QUEUE_DEPTH = 1;
constexpr uint32_t MAX_UINT16 = 65535;

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t GetAivBlockNum()
{
    return static_cast<uint64_t>(GetBlockNum()) * static_cast<uint64_t>(GetTaskRation());
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

__aicore__ inline void CopyRankColumn(GlobalTensor<float> &dst, GlobalTensor<float> &src,
                                      TQue<QuePosition::VECIN, VECTOR_QUEUE_DEPTH> &columnQueue,
                                      uint64_t dstOffset, uint64_t srcOffset)
{
    LocalTensor<float> columnLocal = columnQueue.AllocTensor<float>();

    DataCopyParams copyInParams{
        static_cast<uint16_t>(ROWS),
        static_cast<uint16_t>(FLOAT_BYTES),
        static_cast<uint16_t>((RANKS_PER_INDEX - 1) * FLOAT_BYTES),
        0};
    DataCopyPadParams padParams{
        true,
        0,
        static_cast<uint8_t>(FLOATS_PER_DATA_BLOCK - 1),
        0};
    DataCopyPad(columnLocal, src[srcOffset], copyInParams, padParams);
    columnQueue.EnQue(columnLocal);

    columnLocal = columnQueue.DeQue<float>();
    const uint64_t dstStrideBytes = static_cast<uint64_t>(OUTPUT_COLS - 1) * FLOAT_BYTES;
    if (dstStrideBytes <= MAX_UINT16) {
        DataCopyParams copyOutParams{
            static_cast<uint16_t>(ROWS),
            static_cast<uint16_t>(FLOAT_BYTES),
            static_cast<uint16_t>((FLOATS_PER_DATA_BLOCK - 1) * FLOAT_BYTES),
            static_cast<uint16_t>(dstStrideBytes)};
        DataCopyPad(dst[dstOffset], columnLocal, copyOutParams);
    } else {
        DataCopyParams copyOneParams{1, static_cast<uint16_t>(FLOAT_BYTES), 0, 0};
        for (uint32_t row = 0; row < ROWS; ++row) {
            DataCopyPad(dst[dstOffset + static_cast<uint64_t>(row) * OUTPUT_COLS],
                        columnLocal[row * FLOATS_PER_DATA_BLOCK],
                        copyOneParams);
        }
    }
    columnQueue.FreeTensor(columnLocal);
}

__aicore__ inline bool IsValidUserRank(int32_t userId, int32_t rank, uint32_t userCount)
{
    return userId >= 0 && static_cast<uint32_t>(userId) < userCount &&
           rank >= 0 && static_cast<uint32_t>(rank) < RANKS_PER_INDEX;
}

__aicore__ inline void ProcessCopies(GM_ADDR weightR, GM_ADDR weightI,
                                     GM_ADDR getIdxs, GM_ADDR lens,
                                     GM_ADDR getUserIds, GM_ADDR getUserIdRank,
                                     GlobalTensor<float> &outR, GlobalTensor<float> &outI,
                                     TQue<QuePosition::VECIN, VECTOR_QUEUE_DEPTH> &columnQueue,
                                     uint32_t userCount, uint32_t idxCount,
                                     uint32_t totalUserEntries,
                                     uint64_t blockIdx, uint64_t blockNum)
{
    GlobalTensor<float> weightRGm;
    GlobalTensor<float> weightIGm;
    const uint64_t weightElems = static_cast<uint64_t>(userCount) * INDEX_COUNT * ROWS * RANKS_PER_INDEX;
    weightRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightR), weightElems);
    weightIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightI), weightElems);

    auto idxsGm = reinterpret_cast<__gm__ int32_t *>(getIdxs);
    auto lensGm = reinterpret_cast<__gm__ int32_t *>(lens);
    auto userIdsGm = reinterpret_cast<__gm__ int32_t *>(getUserIds);
    auto ranksGm = reinterpret_cast<__gm__ int32_t *>(getUserIdRank);

    uint64_t taskId = 0;
    uint64_t userOffset = 0;
    for (uint32_t i = 0; i < idxCount; ++i) {
        const int32_t rawLen = lensGm[i];
        const uint64_t currentLen = rawLen > 0 ? static_cast<uint64_t>(rawLen) : 0;
        const uint64_t remaining = userOffset < totalUserEntries ? totalUserEntries - userOffset : 0;
        const uint64_t remainingCols = userOffset < OUTPUT_COLS ? OUTPUT_COLS - userOffset : 0;
        const uint64_t availableLen = MinU64(currentLen, remaining);
        const uint64_t writeLen = MinU64(availableLen, remainingCols);

        const int32_t idxGroup = idxsGm[i];
        const bool validIdxGroup = idxGroup >= 0 && static_cast<uint32_t>(idxGroup) < INDEX_GROUP_COUNT;
        for (uint64_t k = 0; k < writeLen; ++k) {
            const int32_t userId = userIdsGm[userOffset + k];
            const int32_t rank = ranksGm[userOffset + k];
            const bool validUserRank = IsValidUserRank(userId, rank, userCount);

            for (uint32_t offset = 0; offset < INDEX_GROUP_WIDTH; ++offset, ++taskId) {
                if ((taskId % blockNum) != blockIdx || !validIdxGroup || !validUserRank) {
                    continue;
                }

                const uint32_t srcIndex = static_cast<uint32_t>(idxGroup) * INDEX_GROUP_WIDTH + offset;
                const uint32_t dstIndex = i * INDEX_GROUP_WIDTH + offset;
                const uint64_t srcOffset =
                    (((static_cast<uint64_t>(userId) * INDEX_COUNT + srcIndex) * ROWS) * RANKS_PER_INDEX) +
                    static_cast<uint32_t>(rank);
                const uint64_t dstOffset =
                    (static_cast<uint64_t>(dstIndex) * ROWS * OUTPUT_COLS) + userOffset + k;

                CopyRankColumn(outR, weightRGm, columnQueue, dstOffset, srcOffset);
                CopyRankColumn(outI, weightIGm, columnQueue, dstOffset, srcOffset);
            }
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
    TQue<QuePosition::VECIN, VECTOR_QUEUE_DEPTH> columnQueue;
    TBuf<> zeroBuf;
    pipe.InitBuffer(columnQueue, VECTOR_QUEUE_DEPTH, PADDED_COLUMN_BYTES);
    pipe.InitBuffer(zeroBuf, ZERO_CHUNK_ELEMS * FLOAT_BYTES);

    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t blockNum = GetAivBlockNum();
    ClearOutputs(outRGm, outIGm, zeroBuf, outElems, blockIdx, blockNum);
    SyncAll<false>();

    ProcessCopies(weight_r, weight_i, getIdxs, lens, getuserIds, getuserIdRank,
                  outRGm, outIGm, columnQueue, userCount, idxCount,
                  totalUserEntries, blockIdx, blockNum);
#endif
}
