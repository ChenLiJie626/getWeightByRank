#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "common.h"
#include "op_runner.h"

namespace {
constexpr int64_t INDEX_COUNT = 136;
constexpr int64_t ROWS = 256;
constexpr int64_t RANKS = 8;

struct SampleShape {
    int64_t userCount;
    int64_t idxCount;
    int64_t totalUserEntries;
    int64_t totalOutputRows;
};

bool GetFileSize(const std::string &path, size_t &fileSize)
{
    struct stat statBuf;
    if (stat(path.c_str(), &statBuf) != 0 || statBuf.st_size < 0) {
        ERROR_LOG("Get file size failed. path=%s", path.c_str());
        return false;
    }
    fileSize = static_cast<size_t>(statBuf.st_size);
    return true;
}

bool GetElementCount(const std::string &path, size_t elemSize, int64_t &count)
{
    size_t fileSize = 0;
    if (!GetFileSize(path, fileSize) || elemSize == 0 || fileSize % elemSize != 0) {
        ERROR_LOG("Invalid file size. path=%s, elemSize=%zu", path.c_str(), elemSize);
        return false;
    }
    count = static_cast<int64_t>(fileSize / elemSize);
    return true;
}

bool ReadInt32File(const std::string &path, std::vector<int32_t> &values)
{
    int64_t count = 0;
    if (!GetElementCount(path, sizeof(int32_t), count)) {
        return false;
    }
    values.resize(static_cast<size_t>(count));
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        ERROR_LOG("Open int32 file failed. path=%s", path.c_str());
        return false;
    }
    in.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(int32_t)));
    if (!in) {
        ERROR_LOG("Read int32 file failed. path=%s", path.c_str());
        return false;
    }
    return true;
}

int64_t NormalizeRankCount(int32_t rankCount)
{
    if (rankCount <= 0) {
        return 0;
    }
    return rankCount < RANKS ? rankCount : RANKS;
}

bool BuildSampleShape(SampleShape &shape)
{
    int64_t weightElems = 0;
    int64_t weightIElems = 0;
    int64_t idxCount = 0;
    int64_t userIdCount = 0;
    int64_t rankCount = 0;
    if (!GetElementCount("../input/input_weight_r.bin", sizeof(float), weightElems) ||
        !GetElementCount("../input/input_weight_i.bin", sizeof(float), weightIElems) ||
        !GetElementCount("../input/input_get_idxs.bin", sizeof(int32_t), idxCount) ||
        !GetElementCount("../input/input_user_ids.bin", sizeof(int32_t), userIdCount) ||
        !GetElementCount("../input/input_user_ranks.bin", sizeof(int32_t), rankCount)) {
        return false;
    }
    const int64_t elemsPerUser = INDEX_COUNT * RANKS * ROWS;
    if (weightElems <= 0 || weightElems != weightIElems || weightElems % elemsPerUser != 0 ||
        idxCount <= 0 || userIdCount != rankCount) {
        ERROR_LOG("Invalid generated input sizes");
        return false;
    }

    std::vector<int32_t> lens;
    std::vector<int32_t> ranks;
    if (!ReadInt32File("../input/input_lens.bin", lens) ||
        !ReadInt32File("../input/input_user_ranks.bin", ranks)) {
        return false;
    }
    if (static_cast<int64_t>(lens.size()) != idxCount ||
        static_cast<int64_t>(ranks.size()) != userIdCount) {
        ERROR_LOG("Input vector sizes do not match idx/user metadata");
        return false;
    }

    int64_t userOffset = 0;
    int64_t totalOutputRows = 0;
    for (int64_t i = 0; i < idxCount; ++i) {
        const int64_t currentLen = lens[static_cast<size_t>(i)] > 0 ? lens[static_cast<size_t>(i)] : 0;
        if (userOffset + currentLen > userIdCount) {
            ERROR_LOG("lens exceeds user/rank list size");
            return false;
        }
        int64_t currentRows = 0;
        for (int64_t k = 0; k < currentLen; ++k) {
            currentRows += NormalizeRankCount(ranks[static_cast<size_t>(userOffset + k)]);
        }
        totalOutputRows += currentRows * RANKS;
        userOffset += currentLen;
    }
    if (userOffset != userIdCount || totalOutputRows <= 0) {
        ERROR_LOG("Invalid dynamic output rows. consumed=%ld, userIdCount=%ld, totalOutputRows=%ld",
                  userOffset, userIdCount, totalOutputRows);
        return false;
    }

    shape.userCount = weightElems / elemsPerUser;
    shape.idxCount = idxCount;
    shape.totalUserEntries = userIdCount;
    shape.totalOutputRows = totalOutputRows;
    INFO_LOG("Resolved sample shape: userCount=%ld, idxCount=%ld, totalUserEntries=%ld, totalOutputRows=%ld",
             shape.userCount, shape.idxCount, shape.totalUserEntries, shape.totalOutputRows);
    return true;
}
} // namespace

bool g_isDevice = false;
int deviceId = 0;

OperatorDesc CreateOpDesc(const SampleShape &sampleShape)
{
    std::vector<int64_t> weightShape{sampleShape.userCount * INDEX_COUNT, RANKS, ROWS};
    std::vector<int64_t> idxShape{sampleShape.idxCount};
    std::vector<int64_t> userListShape{sampleShape.totalUserEntries};
    std::vector<int64_t> outputShape{sampleShape.totalOutputRows, ROWS};
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(ACL_FLOAT, weightShape.size(), weightShape.data(), format);
    opDesc.AddInputTensorDesc(ACL_FLOAT, weightShape.size(), weightShape.data(), format);
    opDesc.AddInputTensorDesc(ACL_INT32, idxShape.size(), idxShape.data(), format);
    opDesc.AddInputTensorDesc(ACL_INT32, idxShape.size(), idxShape.data(), format);
    opDesc.AddInputTensorDesc(ACL_INT32, userListShape.size(), userListShape.data(), format);
    opDesc.AddInputTensorDesc(ACL_INT32, userListShape.size(), userListShape.data(), format);
    opDesc.AddOutputTensorDesc(ACL_FLOAT, outputShape.size(), outputShape.data(), format);
    opDesc.AddOutputTensorDesc(ACL_FLOAT, outputShape.size(), outputShape.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    if (!ReadFile("../input/input_weight_r.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0)) ||
        !ReadFile("../input/input_weight_i.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1)) ||
        !ReadFile("../input/input_get_idxs.bin", fileSize, runner.GetInputBuffer<void>(2), runner.GetInputSize(2)) ||
        !ReadFile("../input/input_lens.bin", fileSize, runner.GetInputBuffer<void>(3), runner.GetInputSize(3)) ||
        !ReadFile("../input/input_user_ids.bin", fileSize, runner.GetInputBuffer<void>(4), runner.GetInputSize(4)) ||
        !ReadFile("../input/input_user_ranks.bin", fileSize, runner.GetInputBuffer<void>(5), runner.GetInputSize(5))) {
        return false;
    }
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    if (!WriteFile("../output/output_weightout_r.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0)) ||
        !WriteFile("../output/output_weightout_i.bin", runner.GetOutputBuffer<void>(1), runner.GetOutputSize(1))) {
        return false;
    }
    INFO_LOG("Write output success");
    return true;
}

void DestroyResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destroy resource failed");
    } else {
        INFO_LOG("Destroy resource success");
    }
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        } else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    if (aclInit(nullptr) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestroyResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp()
{
    SampleShape sampleShape{};
    if (!BuildSampleShape(sampleShape)) {
        ERROR_LOG("Build sample shape failed");
        return false;
    }
    OperatorDesc opDesc = CreateOpDesc(sampleShape);

    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main()
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestroyResource();
        return FAILED;
    }

    DestroyResource();
    return SUCCESS;
}
