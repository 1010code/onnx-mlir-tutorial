#include <dlfcn.h>
#include <iostream>
#include <memory>

typedef void* (*OmTensorCreateFunc)(void*, int64_t*, int32_t, int32_t);
typedef void (*OmTensorListDestroyFunc)(void*);
typedef void* (*OmTensorListCreateFunc)(void**, int32_t);
typedef void* (*OmTensorListGetOmtByIndexFunc)(void*, int32_t);
typedef void* (*OmTensorGetDataPtrFunc)(void*);

int main() {
    // 動態載入 libcruntime.so
    void* runtimeHandle = dlopen("../onnx-mlir/build/lib/libcruntime.so", RTLD_LAZY);
    if (!runtimeHandle) {
        std::cerr << "Error loading libcruntime.so: " << dlerror() << std::endl;
        return EXIT_FAILURE;
    }

    // 獲取 libcruntime 函數
    auto omTensorCreate = (OmTensorCreateFunc)dlsym(runtimeHandle, "omTensorCreate");
    auto omTensorListDestroy = (OmTensorListDestroyFunc)dlsym(runtimeHandle, "omTensorListDestroy");
    auto omTensorListCreate = (OmTensorListCreateFunc)dlsym(runtimeHandle, "omTensorListCreate");
    auto omTensorListGetOmtByIndex = (OmTensorListGetOmtByIndexFunc)dlsym(runtimeHandle, "omTensorListGetOmtByIndex");
    auto omTensorGetDataPtr = (OmTensorGetDataPtrFunc)dlsym(runtimeHandle, "omTensorGetDataPtr");

    // 檢查是否成功載入
    if (!omTensorCreate || !omTensorListDestroy || !omTensorListCreate || !omTensorListGetOmtByIndex || !omTensorGetDataPtr) {
        std::cerr << "Error loading libcruntime symbols" << std::endl;
        dlclose(runtimeHandle);
        return EXIT_FAILURE;
    }

    // 動態載入 tf_model.so
    void* modelHandle = dlopen("./tf_model.so", RTLD_LAZY);
    if (!modelHandle) {
        std::cerr << "Error loading tf_model.so: " << dlerror() << std::endl;
        dlclose(runtimeHandle);
        return EXIT_FAILURE;
    }

    // 獲取 run_main_graph 函數
    auto run_main_graph = (void* (*)(void*))dlsym(modelHandle, "run_main_graph");
    if (!run_main_graph) {
        std::cerr << "Error loading run_main_graph: " << dlerror() << std::endl;
        dlclose(runtimeHandle);
        dlclose(modelHandle);
        return EXIT_FAILURE;
    }

    // 創建輸入張量
    int64_t shape[] = {1, 4};
    float inputData[] = {6.3, 3.3, 6.0, 2.5};
    void* inputTensor = omTensorCreate(inputData, shape, 2, 1);  // ONNX_TYPE_FLOAT = 1
    void* inputTensorList = omTensorListCreate(&inputTensor, 1);

    // 執行推論
    void* outputTensorList = run_main_graph(inputTensorList);

    // 獲取輸出
    void* outputTensor = omTensorListGetOmtByIndex(outputTensorList, 0);
    float* outputData = (float*)omTensorGetDataPtr(outputTensor);

    std::cout << "模型輸出：";
    for (int i = 0; i < 3; ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;

    // 釋放資源
    omTensorListDestroy(inputTensorList);
    omTensorListDestroy(outputTensorList);

    // 關閉共享庫
    dlclose(runtimeHandle);
    dlclose(modelHandle);

    return 0;
}
