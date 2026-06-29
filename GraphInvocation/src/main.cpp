#include <sys/stat.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge/ge_api.h"
#include "graph/tensor.h"
#include "op_proto.h"

namespace {
constexpr int64_t INDEX_COUNT = 136;
constexpr int64_t RANKS = 8;
constexpr int64_t ROWS = 256;

struct SampleShape {
    int64_t userCount;
    int64_t idxTensorLen;
    int64_t idxCount;
    int64_t userEntryTensorLen;
    int64_t totalOutputRows;
};

bool GetFileSize(const std::string &path, size_t &fileSize)
{
    struct stat statBuf;
    if (stat(path.c_str(), &statBuf) != 0 || statBuf.st_size < 0) {
        std::cerr << "Get file size failed: " << path << std::endl;
        return false;
    }
    fileSize = static_cast<size_t>(statBuf.st_size);
    return true;
}

bool GetElementCount(const std::string &path, size_t elemSize, int64_t &count)
{
    size_t fileSize = 0;
    if (!GetFileSize(path, fileSize) || elemSize == 0 || fileSize % elemSize != 0) {
        std::cerr << "Invalid file size: " << path << std::endl;
        return false;
    }
    count = static_cast<int64_t>(fileSize / elemSize);
    return true;
}

bool ReadFile(const std::string &path, std::vector<uint8_t> &data)
{
    size_t fileSize = 0;
    if (!GetFileSize(path, fileSize)) {
        return false;
    }
    data.resize(fileSize);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Open file failed: " << path << std::endl;
        return false;
    }
    in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!in) {
        std::cerr << "Read file failed: " << path << std::endl;
        return false;
    }
    return true;
}

bool WriteFile(const std::string &path, const uint8_t *data, size_t size)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Open output file failed: " << path << std::endl;
        return false;
    }
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
    return static_cast<bool>(out);
}

ge::TensorDesc MakeDesc(const std::vector<int64_t> &shape, ge::DataType dtype)
{
    return ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, dtype);
}

ge::op::Data MakeData(const char *name, int64_t index, const ge::TensorDesc &desc)
{
    ge::op::Data data(name);
    data.set_attr_index(index);
    data.update_output_desc_y(desc);
    return data;
}

ge::Tensor MakeTensor(const ge::TensorDesc &desc, const std::string &path)
{
    std::vector<uint8_t> data;
    if (!ReadFile(path, data)) {
        return ge::Tensor();
    }
    return ge::Tensor(desc, std::move(data));
}

bool BuildSampleShape(SampleShape &shape)
{
    int64_t weightElems = 0;
    int64_t weightIElems = 0;
    int64_t idxTensorLen = 0;
    int64_t userIdCount = 0;
    int64_t rankCount = 0;
    int64_t totalRowsCount = 0;
    int64_t idxCountElems = 0;
    if (!GetElementCount("../input/input_weight_r.bin", sizeof(float), weightElems) ||
        !GetElementCount("../input/input_weight_i.bin", sizeof(float), weightIElems) ||
        !GetElementCount("../input/input_get_idxs.bin", sizeof(uint32_t), idxTensorLen) ||
        !GetElementCount("../input/input_user_ids.bin", sizeof(uint32_t), userIdCount) ||
        !GetElementCount("../input/input_user_ranks.bin", sizeof(uint32_t), rankCount) ||
        !GetElementCount("../input/input_total_rows.bin", sizeof(uint32_t), totalRowsCount) ||
        !GetElementCount("../input/input_idx_count.bin", sizeof(uint32_t), idxCountElems)) {
        return false;
    }
    const int64_t elemsPerUser = INDEX_COUNT * RANKS * ROWS;
    if (weightElems <= 0 || weightElems != weightIElems || weightElems % elemsPerUser != 0 ||
        idxTensorLen <= 0 || userIdCount != rankCount || totalRowsCount <= 0 || idxCountElems != 1) {
        std::cerr << "Invalid generated input sizes" << std::endl;
        return false;
    }

    std::vector<uint8_t> idxCountBytes;
    if (!ReadFile("../input/input_idx_count.bin", idxCountBytes)) {
        return false;
    }
    const auto idxCount = static_cast<int64_t>(*reinterpret_cast<const uint32_t *>(idxCountBytes.data()));
    if (idxCount <= 0 || idxCount > idxTensorLen) {
        std::cerr << "idxCount input is out of range. idxCount=" << idxCount
                  << ", idxTensorLen=" << idxTensorLen << std::endl;
        return false;
    }

    shape.userCount = weightElems / elemsPerUser;
    shape.idxTensorLen = idxTensorLen;
    shape.idxCount = idxCount;
    shape.userEntryTensorLen = userIdCount;
    shape.totalOutputRows = totalRowsCount;
    std::cout << "Resolved sample input shape: userCount=" << shape.userCount
              << ", idxTensorLen=" << shape.idxTensorLen
              << ", idxCount=" << shape.idxCount
              << ", userEntryTensorLen=" << shape.userEntryTensorLen
              << ", totalOutputRows=" << shape.totalOutputRows << std::endl;
    return true;
}

void PrintShape(const std::string &name, const ge::Tensor &tensor)
{
    const auto dims = tensor.GetTensorDesc().GetShape().GetDims();
    std::cout << name << " inferred shape = [";
    for (size_t i = 0; i < dims.size(); ++i) {
        std::cout << dims[i] << (i + 1 == dims.size() ? "" : ", ");
    }
    std::cout << "], dtype=" << tensor.GetTensorDesc().GetDataType() << std::endl;
}
} // namespace

int main()
{
    SampleShape sampleShape{};
    if (!BuildSampleShape(sampleShape)) {
        return 1;
    }

    const std::vector<int64_t> weightShape{sampleShape.userCount * INDEX_COUNT, RANKS, ROWS};
    const std::vector<int64_t> idxShape{sampleShape.idxTensorLen};
    const std::vector<int64_t> userListShape{sampleShape.userEntryTensorLen};

    const auto weightDesc = MakeDesc(weightShape, ge::DT_FLOAT);
    const auto idxDesc = MakeDesc(idxShape, ge::DT_UINT32);
    const auto lensDesc = MakeDesc(idxShape, ge::DT_UINT32);
    const auto userIdsDesc = MakeDesc(userListShape, ge::DT_UINT32);
    const auto ranksDesc = MakeDesc(userListShape, ge::DT_UINT32);
    const auto totalRowsDesc = MakeDesc({sampleShape.totalOutputRows}, ge::DT_UINT32);
    const auto idxCountDesc = MakeDesc({1}, ge::DT_UINT32);
    const auto outputDesc = MakeDesc({-1, ROWS}, ge::DT_FLOAT);

    auto weightR = MakeData("weight_r", 0, weightDesc);
    auto weightI = MakeData("weight_i", 1, weightDesc);
    auto getIdxs = MakeData("getIdxs", 2, idxDesc);
    auto lens = MakeData("lens", 3, lensDesc);
    auto getuserIds = MakeData("getuserIds", 4, userIdsDesc);
    auto getuserIdRank = MakeData("getuserIdRank", 5, ranksDesc);
    auto totalRows = MakeData("totalRows", 6, totalRowsDesc);
    auto idxCount = MakeData("idxCount", 7, idxCountDesc);

    ge::op::GetWeightByRank op("get_weight_by_rank");
    op.set_input_weight_r(weightR);
    op.set_input_weight_i(weightI);
    op.set_input_getIdxs(getIdxs);
    op.set_input_lens(lens);
    op.set_input_getuserIds(getuserIds);
    op.set_input_getuserIdRank(getuserIdRank);
    op.set_input_totalRows(totalRows);
    op.set_input_idxCount(idxCount);
    op.update_output_desc_weightout_r(outputDesc);
    op.update_output_desc_weightout_i(outputDesc);

    ge::Graph graph("single_get_weight_by_rank");
    graph.SetInputs({weightR, weightI, getIdxs, lens, getuserIds, getuserIdRank, totalRows, idxCount});
    graph.SetOutputs({std::make_pair(op, std::vector<size_t>{0, 1})});

    std::map<std::string, std::string> options = {
        {"ge.exec.deviceId", "0"},
        {"ge.graphRunMode", "0"},
        {"ge.socVersion", "Ascend910B1"}
    };
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed: " << ge::GEGetErrorMsg() << std::endl;
        return 1;
    }

    ge::Session session(options);
    if (session.AddGraph(0, graph) != ge::SUCCESS) {
        std::cerr << "AddGraph failed: " << ge::GEGetErrorMsg() << std::endl;
        ge::GEFinalize();
        return 1;
    }

    std::vector<ge::Tensor> inputs;
    inputs.emplace_back(MakeTensor(weightDesc, "../input/input_weight_r.bin"));
    inputs.emplace_back(MakeTensor(weightDesc, "../input/input_weight_i.bin"));
    inputs.emplace_back(MakeTensor(idxDesc, "../input/input_get_idxs.bin"));
    inputs.emplace_back(MakeTensor(lensDesc, "../input/input_lens.bin"));
    inputs.emplace_back(MakeTensor(userIdsDesc, "../input/input_user_ids.bin"));
    inputs.emplace_back(MakeTensor(ranksDesc, "../input/input_user_ranks.bin"));
    inputs.emplace_back(MakeTensor(totalRowsDesc, "../input/input_total_rows.bin"));
    inputs.emplace_back(MakeTensor(idxCountDesc, "../input/input_idx_count.bin"));

    if (session.BuildGraph(0, inputs) != ge::SUCCESS) {
        std::cerr << "BuildGraph failed: " << ge::GEGetErrorMsg() << std::endl;
        ge::GEFinalize();
        return 1;
    }
    std::cout << "BuildGraph success" << std::endl;

    std::vector<ge::Tensor> outputs;
    const ge::Status ret = session.RunGraph(0, inputs, outputs);
    if (ret != ge::SUCCESS) {
        std::cerr << "RunGraph failed: " << ge::GEGetErrorMsg() << std::endl;
        ge::GEFinalize();
        return 1;
    }

    std::cout << "RunGraph success, output count=" << outputs.size() << std::endl;
    for (size_t i = 0; i < outputs.size(); ++i) {
        PrintShape("output[" + std::to_string(i) + "]", outputs[i]);
    }

    if (outputs.size() != 2 ||
        !WriteFile("../output/output_weightout_r.bin", outputs[0].GetData(), outputs[0].GetSize()) ||
        !WriteFile("../output/output_weightout_i.bin", outputs[1].GetData(), outputs[1].GetSize())) {
        std::cerr << "Write graph outputs failed" << std::endl;
        ge::GEFinalize();
        return 1;
    }

    if (ge::GEFinalize() != ge::SUCCESS) {
        std::cerr << "GEFinalize failed" << std::endl;
        return 1;
    }
    return 0;
}
