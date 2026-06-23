#ifndef GET_WEIGHT_BY_RANK_TILING_H
#define GET_WEIGHT_BY_RANK_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace GetWeightByRankConst {
constexpr int32_t INDEX_COUNT = 136;
constexpr int32_t ROWS = 256;
constexpr int32_t RANKS_PER_INDEX = 8;
constexpr int32_t INDEX_GROUP_WIDTH = 8;
constexpr int32_t OUTPUT_COLS = 8;
constexpr int32_t DEFAULT_BLOCK_DIM = 16;
} // namespace GetWeightByRankConst

namespace optiling {
BEGIN_TILING_DATA_DEF(GetWeightByRankTilingData)
TILING_DATA_FIELD_DEF(uint32_t, userCount);
TILING_DATA_FIELD_DEF(uint32_t, idxCount);
TILING_DATA_FIELD_DEF(uint32_t, totalUserEntries);
TILING_DATA_FIELD_DEF(uint32_t, totalOutputRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GetWeightByRank, GetWeightByRankTilingData)
} // namespace optiling

#endif // GET_WEIGHT_BY_RANK_TILING_H
