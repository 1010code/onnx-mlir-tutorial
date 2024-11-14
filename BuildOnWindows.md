## ONNX-MLIR Windows 安裝
ONNX-MLIR 支援透過 MSVC 使用CMake 建置。需要有一個 Visual Studio 編譯器。 VS 的最低版本為Visual Studio Enterprise 2019。
開始工具列搜尋  Developer Command Prompt for VS 2019 開啟終端機。

開啟CMD後進入 conda 環境。

```
REM 設置 Miniconda 的安裝路徑
SET CONDA_PATH=D:\software\miniconda3

REM 初始化 Conda 環境
CALL "%CONDA_PATH%\Scripts\activate.bat" "%CONDA_PATH%"
```

這裡我們透過 conda 指令建立一個新的 Python 環境。

```
REM make sure to start with a fresh environment
conda env remove -n onnx-mlir-venv
REM create the conda environment with build dependency
conda create -n onnx-mlir-venv -c conda-forge ^
    "llvmdev>=15" ^
    "cmake>=3.24" ^
    git ^
    python=3.11
```

### 下載 Protobuf 專案
ONNX-MLIR 依賴於 Protobuf。首先，下載 Protobuf 專案的特定版本並編譯。

```
REM Check out protobuf v21.12
set protobuf_version=21.12
git clone -b v%protobuf_version% --recursive https://github.com/protocolbuffers/protobuf.git

set root_dir=%cd%
md protobuf_build
cd protobuf_build
call cmake %root_dir%\protobuf\cmake -G "Ninja" ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\protobuf_install" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -Dprotobuf_BUILD_EXAMPLES=OFF ^
   -Dprotobuf_BUILD_SHARED_LIBS=OFF ^
   -Dprotobuf_BUILD_TESTS=OFF ^
   -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ^
   -Dprotobuf_WITH_ZLIB=OFF

call cmake --build . --config Release
call cmake --build . --config Release --target install
```

在為 onnx-mlir 執行 CMake 之前，請確保此 protobuf 的 bin 目錄被設置在環境變數中：

```sh
set PATH=%root_dir%\protobuf_install\bin;%PATH%
```




### 下載 LLVM 專案
ONNX-MLIR 依賴於 LLVM 和 MLIR。這裡，下載 LLVM 專案的特定版本並編譯。

```sh
cd ../
git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 00128a20eec27246719d73ba427bf821883b00b4 && cd ..
```

在 llvm-project 目錄中，創建一個新的 build 目錄，然後使用 CMake 和 Ninja 進行編譯。

```sh
set root_dir=%cd%
md llvm-project\build
cd llvm-project\build

call cmake %root_dir%\llvm-project\llvm -G "Ninja" ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\llvm-project\build\install" ^
   -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" ^
   -DLLVM_TARGETS_TO_BUILD="host" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -DLLVM_ENABLE_ASSERTIONS=ON ^
   -DLLVM_ENABLE_RTTI=ON ^
   -DLLVM_ENABLE_ZLIB=OFF ^
   -DLLVM_INSTALL_UTILS=ON ^
   -DENABLE_LIBOMPTARGET=OFF ^
   -DLLVM_ENABLE_LIBEDIT=OFF

call cmake --build . --config Release
call cmake --build . --config Release --target install
call cmake --build . --config Release --target check-mlir
```


### 下載 ONNX-MLIR 專案


```sh
cd ../
git clone --recursive https://github.com/onnx/onnx-mlir.git

set root_dir=%cd%

md onnx-mlir\build
cd onnx-mlir\build
call cmake %root_dir%\onnx-mlir -G "Ninja" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -DCMAKE_PREFIX_PATH=%root_dir%\protobuf_install ^
   -DLLVM_EXTERNAL_LIT=%lit_path% ^
   -DLLVM_LIT_ARGS=-v ^
   -DMLIR_DIR=%root_dir%\llvm-project\build\lib\cmake\mlir ^
   -DONNX_MLIR_ENABLE_STABLEHLO=OFF ^
   -DONNX_MLIR_ENABLE_WERROR=ON

call cmake --build . --config Release
```