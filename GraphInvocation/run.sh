#!/bin/bash
CURRENT_DIR=$(
    cd "$(dirname "${BASH_SOURCE:-$0}")"
    pwd
)

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

source "$_ASCEND_INSTALL_PATH/bin/setenv.bash"
export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/lib64
export ASCEND_OPP_PATH=$_ASCEND_INSTALL_PATH/opp
export ASCEND_CUSTOM_OPP_PATH=$_ASCEND_INSTALL_PATH/opp/vendors/customize
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_proto/lib/linux/aarch64:$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$LD_LIBRARY_PATH

function main {
    rm -rf "$HOME"/ascend/log/*
    rm -rf "$CURRENT_DIR"/input "$CURRENT_DIR"/output
    mkdir -p "$CURRENT_DIR"/input "$CURRENT_DIR"/output

    cd "$CURRENT_DIR"
    python3 ../AclNNInvocation/scripts/gen_data.py
    if [ $? -ne 0 ]; then
        echo "ERROR: generate input data failed!"
        return 1
    fi
    echo "INFO: generate input data success!"

    rm -rf build
    mkdir -p build
    cd build
    cmake ../src -DCMAKE_SKIP_RPATH=TRUE
    if [ $? -ne 0 ]; then
        echo "ERROR: cmake failed!"
        return 1
    fi
    echo "INFO: cmake success!"
    make
    if [ $? -ne 0 ]; then
        echo "ERROR: make failed!"
        return 1
    fi
    echo "INFO: make success!"

    export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/lib64:$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_proto/lib/linux/aarch64:$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    cd "$CURRENT_DIR/output"
    echo "INFO: execute graph!"
    ./execute_get_weight_by_rank_graph
    if [ $? -ne 0 ]; then
        echo "ERROR: graph executable run failed!"
        return 1
    fi
    echo "INFO: graph executable run success!"

    cd "$CURRENT_DIR"
    python3 ../AclNNInvocation/scripts/verify_result.py
}

main
