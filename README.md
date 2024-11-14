

# 使用 ONNX-MLIR 優化 ONNX 模型並在 C++ 中進行推論
隨著人工智慧和機器學習應用的快速發展，越來越多的框架和工具支持將訓練好的模型進行優化並部署在不同的硬體環境中。ONNX-MLIR 是其中一個專門用於 ONNX 模型的優化和編譯工具，它可以將 ONNX 格式的機器學習模型轉換為高度優化的可執行文件（如 .so 或 .dll 動態庫）。這樣的轉換使得模型可以在目標硬體上以較高的效率運行，並且利用 LLVM 和 MLIR 的優化功能，進一步減少了運行時的延遲和計算資源的消耗。

ONNX（Open Neural Network Exchange）作為開放的中間格式，使得不同機器學習框架之間的模型可以互相轉換和運行。ONNX-MLIR 就是基於這一格式的模型編譯工具，能夠處理符合 ONNX 標準的模型，將其轉換為不同平台的動態庫，使得在嵌入式設備、伺服器等多種環境中進行高效的推理成為可能。

在本篇文章中，我們將使用 ONNX-MLIR 來展示如何從頭到尾將一個簡單的 DNN 模型進行轉換和推論。本次範例將基於鳶尾花（Iris）分類模型進行展示，並分為以下三個步驟：

- **建立並保存 ONNX 模型檔案**：我們將使用一個簡單的 DNN 模型來訓練鳶尾花資料集，並將該模型保存為 ONNX 格式的檔案。
- **使用 ONNX-MLIR 轉換模型為共享庫**：接下來，我們會使用 ONNX-MLIR 工具將生成的 ONNX 模型轉換為可執行的共享庫（.so 文件），該文件包含了模型的推理邏輯和執行所需的函數。
- **撰寫 C++ 程式進行推論**：最後，我們將撰寫一個 C++ 程式，透過 OnnxMlirRuntime 提供的 C++ API 載入 .so 動態庫並執行模型推論。


## ONNX-MLIR 安裝
在這篇文章中，我們將以 macOS 作為範例，介紹如何安裝和配置 ONNX-MLIR，以便您能夠在 macOS 環境下使用這個工具來編譯 ONNX 模型。ONNX-MLIR 提供了多平台的支持，若您使用的是其他系統，可以參考官方安裝指南中的[Linux/macOS](https://github.com/onnx/onnx-mlir/blob/main/docs/BuildOnLinuxOSX.md)或[Windows](https://github.com/onnx/onnx-mlir/blob/main/docs/BuildOnWindows.md)指引來進行安裝。

### 安裝必要的工具和庫
使用 Homebrew 安裝 CMake、Ninja 和其他必要的工具。

```sh
brew install cmake ninja
```

Homebrew 是 macOS 上的軟體包管理工具。如果尚未安裝，請在終端機中執行以下命令：

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 下載 LLVM 專案
ONNX-MLIR 依賴於 LLVM 和 MLIR。首先，下載 LLVM 專案的特定版本並編譯。

```sh
git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 00128a20eec27246719d73ba427bf821883b00b4 && cd ..
```

在 llvm-project 目錄中，創建一個新的 build 目錄，然後使用 CMake 和 Ninja 進行編譯。

```sh
mkdir llvm-project/build
cd llvm-project/build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DENABLE_LIBOMPTARGET=OFF \
   -DLLVM_ENABLE_LIBEDIT=OFF

cmake --build . -- ${MAKEFLAGS}
cmake --build . --target check-mlir
```

### 下載 ONNX-MLIR
返回上一層目錄，然後下載 ONNX-MLIR 的 GitHub 儲存庫。

```sh
cd ..
git clone --recursive https://github.com/onnx/onnx-mlir.git
```

進入 onnx-mlir 目錄，創建一個新的 build 目錄，然後使用 CMake 和 Ninja 進行編譯。

```sh
# MLIR_DIR must be set with cmake option now
MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir
mkdir onnx-mlir/build && cd onnx-mlir/build
if [[ -z "$pythonLocation" ]]; then
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
else
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DPython3_ROOT_DIR=$pythonLocation \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
fi
cmake --build .

# Run lit tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
```

在安裝 ONNX-MLIR 的過程中，確保使用包含 Python 的環境非常重要。建議使用 conda 來建立和管理開發環境。

## 1. 建立並保存 ONNX 模型檔案
以下是一個使用 TensorFlow 建立鳶尾花（Iris）分類模型並將其導出為 ONNX 格式的範例。該模型使用簡單的全連接層來進行分類，並轉換為 ONNX 格式，方便在 ONNX-MLIR 或其他 ONNX 支持的推理引擎上運行。

### 1.1 安裝必要的套件
如果尚未安裝 tensorflow 和 tf2onnx，可以使用以下命令安裝：

```sh
pip install tensorflow tf2onnx
```

### 1.2 建立並訓練 TensorFlow 模型
以下程式碼將建立一個簡單的神經網絡來分類鳶尾花數據集，並將其導出為 ONNX 格式。

```py
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 載入鳶尾花資料集
iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,)),  # 4 個特徵
    tf.keras.layers.Dense(10, activation='relu'),  # 隱藏層
    tf.keras.layers.Dense(3, activation='softmax') # 輸出層，3 個分類
])

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=0)

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"模型準確率: {accuracy:.2f}")
```

### 1.3 將模型轉換為 ONNX 格式
使用 tf2onnx 將訓練好的 TensorFlow 模型轉換為 ONNX 格式：

```py
import tf2onnx

# 將 Keras 模型轉換為 ONNX 格式
spec = (tf.TensorSpec((None, 4), tf.float32, name="float_input"),)  # 定義輸入規範
output_path = "tf_model.onnx"  # 輸出 ONNX 模型的路徑

# 轉換模型
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"ONNX 模型已保存至 {output_path}")
```

執行此程式碼後，會在當前目錄下生成一個名為 tf_model.onnx 的文件。這個文件就是已經轉換好的鳶尾花分類模型的 ONNX 格式，可以用於後續的 ONNX-MLIR 編譯或其他 ONNX 支援的推理工具上。

## 2. 使用 ONNX-MLIR 轉換模型為共享庫
以下是如何使用 ONNX-MLIR 將 tf.onnx 模型轉換為共享庫（.so 文件）的步驟。

### 2.1 將 ONNX 模型轉換為共享庫
使用 onnx-mlir 將 tf_model.onnx 模型轉換為共享庫（.so 文件）。執行以下命令：

```sh
onnx-mlir --EmitLib tf_model.onnx
```

> 執行完成後，您應該會在當前目錄下得到一個名為 tf_model.so 的文件。

這條命令會進行以下操作：

- **解析模型**：ONNX-MLIR 會讀取 tf_model.onnx 模型文件，將其轉換為中間表示(LLMIR)，並進行優化。
- **生成共享庫**：ONNX-MLIR 使用 LLVM 從中間表示生成目標平台的二進制文件（動態庫）。在 macOS 和 Linux 系統上生成的文件為 .so 格式，在 Windows 系統上則為 .dll 格式。

輸出結果：
```
[1/6] Mon Nov 11 21:34:56 2024 (0s) Importing ONNX Model to MLIR Module from "tf_model.onnx"
[2/6] Mon Nov 11 21:34:56 2024 (0s) Compiling and Optimizing MLIR Module
[3/6] Mon Nov 11 21:34:56 2024 (0s) Translating MLIR Module to LLVM and Generating LLVM Optimized Bitcode
[4/6] Mon Nov 11 21:34:56 2024 (0s) Generating Object from LLVM Bitcode
[5/6] Mon Nov 11 21:34:56 2024 (0s) Linking and Generating the Output Shared Library
[6/6] Mon Nov 11 21:34:56 2024 (0s) Compilation completed
```

生成的 tf_model.so 檔案現在可以用作獨立的模型推理庫，並可以使用 OnnxMlirRuntime 進行載入和推論。我們可以進一步撰寫 C++ 程式，使用 OnnxMlirRuntime 函數載入該共享庫並執行推論。

## 3. 撰寫 C++ 程式進行推論
### 3.1 撰寫 C++ 程式
這段 C++ 程式碼的目的是載入使用 ONNX-MLIR 編譯生成的機器學習模型（.so 動態庫），並進行模型推論。首先，它創建並初始化模型所需的輸入張量，然後調用模型的推理函數 run_main_graph 來執行計算。推論完成後，程式從輸出張量中提取數據並顯示結果。最後，釋放張量和張量列表的資源以避免記憶體洩漏。該程式展示了如何使用 OnnxMlirRuntime 在 C++ 中進行高效的機器學習推論。

```c

```

### 3.2 編譯程式
要編譯上述 C++ 程式碼，除了確保 OnnxMlirRuntime 和生成的 .so 模型文件（例如 iris_model.so）在當前目錄，還需要指定 OnnxMlirRuntime.h 頭文件的路徑。
因為程式中引用了 OnnxMlirRuntime.h，因此需要使用 -I 選項指定頭文件的路徑。如果您將 ONNX-MLIR 的安裝位置放在 onnx-mlir/include 目錄中，可以使用以下命令進行編譯：

```sh
g++ --std=c++17 inference.cpp -L. -liris_model -lOnnxMlirRuntime -o main -Ionnx-mlir/include -ldl -lpthread
```

### 3.3 執行推論
編譯成功後，您可以使用以下命令運行生成的可執行文件 main 來執行推論：

```
./main
```

執行後，程式將輸出模型的推理結果。在本範例中模型輸出的格式為 `[1, 3]` 的張量，表示三個分類的機率，將看到如下類似的結果：

```
模型輸出：0.00611059 0.276536 0.717354
```


## 小結
`ONNX-MLIR` 是一個將 ONNX 模型轉換為高效可執行文件（如 `.so` 動態庫）的工具。它的工作流程如下：

1. **輸入 ONNX 模型**：將機器學習模型保存為 ONNX 格式（`.onnx` 文件），該格式是跨框架和跨平台的標準格式，允許在多種工具之間互操作。
  
2. **編譯和優化**：ONNX-MLIR 使用 LLVM 和 MLIR 的優化功能，將 ONNX 模型轉換為 LLVM 中間表示（IR），並進行多層優化。這些優化包括圖優化、算子融合等，以便在目標硬體上更高效地運行模型。

3. **生成共享庫 (`.so` 文件)**：最終，ONNX-MLIR 將優化後的 LLVM IR 編譯為目標平台的可執行二進制文件（在 Linux 和 macOS 上為 `.so` 動態庫；在 Windows 上則是 `.dll` 文件）。

## `.so` 文件的作用和用途

生成的 `.so` 文件是一個**動態連結庫**（Shared Library），其中包含了模型的推理邏輯和執行所需的算子實現。這個動態庫的用途如下：

- **獨立運行的推理庫**：生成的 `.so` 文件可以在支持該二進制格式的系統上（例如 Linux 的 `.so` 文件可以在 Linux 上運行）直接加載，進行模型推理。

- **使用 OnnxMlirRuntime 進行推理**：ONNX-MLIR 提供了 `OnnxMlirRuntime` 運行時庫，用於管理和執行生成的 `.so` 文件。`OnnxMlirRuntime` 提供了數據結構（如 `OMTensor`）來管理輸入輸出張量，以及 `run_main_graph` 等函數來加載 `.so` 文件並進行推理。

- **C++/Python 接口**：可以使用 C++ 或 Python 程式碼來加載和調用 `.so` 文件中的推理函數。通過 `OnnxMlirRuntime` 的接口，您可以用 `.so` 文件對模型進行推理。


## Reference
- [ONNX模型檔->可執行檔C Runtime通路詳細實作方法](https://zhuanlan.zhihu.com/p/356735007)
- [ONNX 介紹](https://blog.trendmicro.com.tw/?p=82293)


ONNX模型文件->可执行文件 C Runtime通路 详细实现方法 https://zhuanlan.zhihu.com/p/356735007
怎样去学习mlir这套框架？ https://www.zhihu.com/question/435109274
2023年后，AI 还有什么研究方向有前景？ https://www.zhihu.com/question/591140366