#include <dlfcn.h>
#include "OnnxMlirRuntime.h"  // 引入 libcruntime 頭文件
#include <iostream>
#include <memory>

// 定義函數指針類型
typedef OMTensorList* (*RunMainGraphFunc)(OMTensorList*);

int main() {
    // 1. 動態載入共享庫
    const char* libPath = "./tf_model.so";  // 指定共享庫路徑
    void* handle = dlopen(libPath, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading shared library: " << dlerror() << std::endl;
        return EXIT_FAILURE;
    }

    // 2. 獲取共享庫中的函數指針
    RunMainGraphFunc run_main_graph = (RunMainGraphFunc)dlsym(handle, "run_main_graph");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Error loading symbol 'run_main_graph': " << dlsym_error << std::endl;
        dlclose(handle);
        return EXIT_FAILURE;
    }

    // 3. 定義輸入張量
    int64_t shape[] = {1, 4};  // 形狀為 [1, 4]
    float inputData[] = {6.3, 3.3, 6.0, 2.5};  // 輸入數據

    // 使用 libcruntime API 創建輸入張量
    OMTensor* inputTensor = omTensorCreate(inputData, shape, 2, ONNX_TYPE_FLOAT);

    // 創建張量列表
    OMTensorList* inputTensorList = omTensorListCreate(&inputTensor, 1);

    // 4. 調用模型推論函數
    OMTensorList* outputTensorList = run_main_graph(inputTensorList);

    // 5. 獲取推論結果
    OMTensor* outputTensor = omTensorListGetOmtByIndex(outputTensorList, 0);
    float* outputData = (float*)omTensorGetDataPtr(outputTensor);

    // 假設輸出為 [1, 3] 的張量，表示三個分類的概率
    std::cout << "模型輸出：";
    for (int i = 0; i < 3; ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;

    // 6. 清理資源
    omTensorListDestroy(inputTensorList);
    omTensorListDestroy(outputTensorList);

    // 7. 關閉共享庫
    dlclose(handle);

    return 0;
}
