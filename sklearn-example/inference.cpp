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
