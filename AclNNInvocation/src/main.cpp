#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "common.h"
#include "op_runner.h"

namespace {
constexpr int64_t USER_COUNT = 30;
constexpr int64_t INDEX_COUNT = 136;
constexpr int64_t ROWS = 256;
constexpr int64_t RANKS = 8;
constexpr int64_t IDX_COUNT = 17;
constexpr int64_t TOTAL_USER_ENTRIES = 34;
constexpr int64_t OUT_ROWS = 2;
} // namespace

bool g_isDevice = false;
int deviceId = 0;

OperatorDesc CreateOpDesc()
{
    std::vector<int64_t> weightShape{USER_COUNT, INDEX_COUNT, RANKS, ROWS};
    std::vector<int64_t> idxShape{IDX_COUNT};
    std::vector<int64_t> userListShape{TOTAL_USER_ENTRIES};
    std::vector<int64_t> outputShape{IDX_COUNT * RANKS, OUT_ROWS, ROWS};
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
    OperatorDesc opDesc = CreateOpDesc();

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
