
## ONNX-MLIR
ONNX-MLIR 是一個強大的編譯器工具，可將 ONNX 模型直接編譯為針對目標硬體架構優化的機器碼。透過 ONNX-MLIR，開發者可以將機器學習模型高效地部署在各種平台上，包括嵌入式系統。雖然它不生成 C/C++ 原始碼，但生成的二進制程式碼可直接在目標設備上運行，無需額外的編譯步驟。

具體而言，ONNX-MLIR 的工作流程如下：

- **ONNX 方言（Dialect）**：ONNX-MLIR 定義了一個 ONNX 方言，用於在 MLIR 中表示 ONNX 模型的操作和結構。 
- **模型降階（Lowering）**：將 ONNX 方言中的操作逐步降階為更低層次的 MLIR 表示，最終轉換為 LLVM IR。 
- **機器碼生成**：利用 LLVM 的生成器，將 LLVM IR 編譯為特定硬體架構的機器碼，生成可執行的二進位文件或庫。


ONNX-MLIR 相當於構建了一個針對 ONNX 模型的「編譯器」，它使用了 MLIR 的多層次表示能力來進行深度優化，並將最終的模型轉換為 LLVM IR，交給 LLVM 的後端生成特定硬體架構的機器碼。因此，ONNX-MLIR 利用 LLVM 的模組化架構，使得 ONNX 模型可以在不同的硬體架構上高效運行。

- 前端（Frontend）——解析 ONNX 模型：
    - 在 ONNX-MLIR 中，前端負責解析 ONNX 模型，將其轉換為 MLIR 的 ONNX 方言（Dialect）表示。這一步相當於 LLVM 編譯器中的前端，將高階語言（在此為 ONNX 模型）轉換為中間表示（IR）。
    - 這一層是將 ONNX 模型的操作和結構轉換為 MLIR 表示，使得模型可以進入優化流程。
- 優化器（Optimizer）——使用 MLIR 進行多層次優化：
    - ONNX-MLIR 使用 MLIR 進行優化，這類似於 LLVM 優化器的功能，但在 MLIR 中分為多層次的方言優化。例如，ONNX 方言可以逐步被降階到 Krnl 方言、Affine 方言，最終轉換為 LLVM 方言。
    - 在這個過程中，可以進行針對 ONNX 模型的特定優化，這些優化包括計算圖優化、常量折疊、循環展開等，使生成的代碼在後續步驟中能夠更高效運行。
- 後端（Backend）——將 LLVM IR 轉換為機器碼：
    - 經過多層次優化後，MLIR 的方言會最終轉換為 LLVM IR。這時候，ONNX-MLIR 會將優化過的 LLVM IR 傳給 LLVM 的後端。
    - LLVM 後端負責將 LLVM IR 編譯為針對特定硬體架構（如 x86、ARM 等）的機器碼，這樣就完成了 ONNX 模型的編譯過程。


## 使用 ONNX-MLIR
成功安裝 ONNX-MLIR 之後，透過一個實作範例將一個鳶尾花朵邏輯迴歸分類器onnx轉換輸出動態連結庫(so)並使用C++進行推論。

### 準備鳶尾花邏輯回歸分類器的 ONNX 模型
使用 Python 和 scikit-learn 來建立並輸出ONNX模型

```py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 載入鳶尾花資料集
iris = load_iris()
X, y = iris.data, iris.target

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 建立模型Pipeline：標準化 + 邏輯迴歸
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 訓練模型
model.fit(X_train, y_train)
```

ONNX-MLIR 主要用於將 ONNX 格式的模型轉換為高效的機器碼，以便在不同硬體平台上執行。然而，ONNX-MLIR 對於由 scikit-learn 生成的 ONNX 模型的支持可能有限=>[參考](https://github.com/onnx/onnx-mlir/issues/2519)。這裡採用 Hummingbird 開源庫，將傳統機器學習模型轉換為張量計算PyTorch框架的等效模型。
```py
from hummingbird.ml import convert

# 將 scikit-learn 模型轉換為 ONNX 格式
hb_model = convert(onnx_model, 'onnx')

# 保存轉換後的 ONNX 模型
hb_model.save('iris_logistic_regression_torch.onnx')
```

### 使用 ONNX-MLIR 將模型編譯編譯為動態連結庫
使用 onnx-mlir 指令解析 onnx 文件，並輸出動態連結庫。

```
./onnx-mlir --EmitLib deploy_model.onnx 
```

生成的 xxx.so 的用途。這是一個動態鏈接庫，包含了模型的編譯後代碼，可以在運行時被其他程式加載和調用。我們可以在 C++、Python 等語言的程式中，載入這個共享庫，使用 ONNX-MLIR 提供的運行時 API，對模型進行推論。

### 編寫 C++ 程式以載入並執行模型

```c
// inference.cpp
#include "OnnxMlirRuntime.h"
#include <iostream>

// 声明模型函数，函数名可能根据模型而异，常见为 'run_main_graph'
extern "C" OMTensorList* run_main_graph(OMTensorList*);

int main() {
    // 定义输入张量的形状，例如 [1, 4]
    int64_t shape[] = {1, 4};
    // 创建输入数据，例如鸢尾花的四个特征值
    float inputData[] = {6.3, 3.3, 6. , 2.5};

    // 创建输入张量
    OMTensor* inputTensor = omTensorCreate(inputData, shape, 2, ONNX_TYPE_FLOAT);

    // 创建包含输入张量的张量列表
    OMTensorList* inputTensorList = omTensorListCreate(&inputTensor, 1);

    // 调用模型函数进行推理
    OMTensorList* outputTensorList = run_main_graph(inputTensorList);

    // 获取输出张量
    OMTensor* outputTensor = omTensorListGetOmtByIndex(outputTensorList, 1);

    // 获取输出数据
    float* outputData = (float*)omTensorGetDataPtr(outputTensor);

    // 假设输出为 [1, 3] 的张量，表示三个类别的概率
    std::cout << "模型输出：";
    for (int i = 0; i < 3; ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;

    // 释放张量列表
    omTensorListDestroy(inputTensorList);
    omTensorListDestroy(outputTensorList);

    return 0;
}
```

接著輸入以下指令透過 g++ 執行編譯產生執行檔。

```sh
g++ --std=c++17 inference.cpp deploy_model.so -o main -I onnx-mlir/include
```

編譯好之後即可執行並查看預測結果。
```sh
./main
```


## 注意事項
- 編譯過程中必須使用 conda 進入虛擬環境
