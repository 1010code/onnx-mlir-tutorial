#include "OnnxMlirRuntime.h"  // 引入 OnnxMlirRuntime 頭文件
#include <iostream>

// 聲明模型函數 'run_main_graph'
extern "C" OMTensorList* run_main_graph(OMTensorList*);

int main() {
    // 定義輸入張量的形狀，例如 [1, 4] 表示一個樣本，四個特徵
    int64_t shape[] = {1, 4};
    // 建立輸入數據，例如鳶尾花的四個特徵值
    float inputData[] = {6.3, 3.3, 6.0, 2.5};

    // 建立輸入張量
    OMTensor* inputTensor = omTensorCreate(inputData, shape, 2, ONNX_TYPE_FLOAT);

    // 建立包含輸入張量的張量列表
    OMTensorList* inputTensorList = omTensorListCreate(&inputTensor, 1);

    // 調用模型函數進行推論
    OMTensorList* outputTensorList = run_main_graph(inputTensorList);

    // 取得輸出張量
    OMTensor* outputTensor = omTensorListGetOmtByIndex(outputTensorList, 0);

    // 取得輸出數據指標
    float* outputData = (float*)omTensorGetDataPtr(outputTensor);

    // 假設輸出為 [1, 3] 的張量，表示三個分類的概率
    std::cout << "模型輸出：";
    for (int i = 0; i < 3; ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;

    // 釋放張量列表資源
    omTensorListDestroy(inputTensorList);
    omTensorListDestroy(outputTensorList);

    return 0;
}
