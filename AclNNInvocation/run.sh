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
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

function main {
    rm -rf "$HOME"/ascend/log/*
    rm -f "$CURRENT_DIR"/input/*.bin
    rm -f "$CURRENT_DIR"/output/*.bin

    cd "$CURRENT_DIR"
    python3 scripts/gen_data.py
    if [ $? -ne 0 ]; then
        echo "ERROR: generate input data failed!"
        return 1
    fi
    echo "INFO: generate input data success!"

    cd "$CURRENT_DIR"
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

    export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_api/lib:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    cd "$CURRENT_DIR/output"
    echo "INFO: execute op!"
    ./execute_get_weight_by_rank_op
    if [ $? -ne 0 ]; then
        echo "ERROR: acl executable run failed! please check your project!"
        return 1
    fi
    echo "INFO: acl executable run success!"

    cd "$CURRENT_DIR"
    python3 scripts/verify_result.py
    if [ $? -ne 0 ]; then
        echo "ERROR: verify result failed!"
        return 1
    fi
}

main
