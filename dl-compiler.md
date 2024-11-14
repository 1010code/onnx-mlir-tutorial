# ONNX-MLIR 與 TVM：機器學習模型編譯器的比較與分析

在深度學習模型的部署中，如何高效地編譯與優化模型以便在不同硬體上進行推理是一個至關重要的問題。**ONNX-MLIR** 和 **TVM** 是兩個在此領域備受矚目的工具，儘管它們有一些相似之處，但在設計目標、應用場景和優化能力上也存在顯著差異。本篇文章將深入比較 ONNX-MLIR 和 TVM，幫助讀者更好地理解它們的特性及應用場景。

## ONNX-MLIR 和 TVM 的比較

ONNX-MLIR 和 TVM 都是將機器學習模型編譯和優化為高效可執行文件的工具，但它們的設計目標和實現方式有所不同。以下是它們的相同點和差異，幫助您了解這兩個工具的核心功能和適用場景。

#### 相同點

1. **模型編譯和優化**

   - 兩者的主要目的是將機器學習模型轉換為高效可執行的代碼，以便在不同平台上進行推理。
   - 它們都支持模型優化，例如算子融合和內存優化，以提高模型在特定硬件上的推理速度。

2. **跨平台支持**

   - 兩者都支持在多種硬件平台上運行，包括 CPU 和某些加速硬件（如 GPU 和嵌入式設備）。
   - 它們都基於 LLVM，因此擁有廣泛的架構支持，如 x86\_64、ARM 和其他特殊硬件平台。

3. **支持 ONNX**

   - ONNX-MLIR 和 TVM 都能處理 ONNX 格式的模型。ONNX 作為開放的交換格式，使得這些工具能夠跨框架地運行模型（例如從 PyTorch、TensorFlow 輸出到 ONNX 格式，然後用 ONNX-MLIR 或 TVM 編譯）。

#### 不同點

1. **設計目標和主要應用場景**

   - **ONNX-MLIR**：專注於 **ONNX 模型的優化和部署**。它是專門針對 ONNX 格式設計的編譯器，並利用 MLIR（多級中間表示）來優化計算圖，最後通過 LLVM 生成機器代碼。ONNX-MLIR 的使用更直接適合已經轉換為 ONNX 的模型。
   - **TVM**：作為 **通用的深度學習編譯器**，支持多種前端格式（例如 TensorFlow、PyTorch 以及 ONNX），並且強調在各種硬件（如 CPU、GPU、TPU 等）上的自動優化和硬件特定的代碼生成。TVM 更適合於多樣化的模型和硬件環境，並且可以自動調整性能以適應特定設備。

2. **支持的模型格式**

   - **ONNX-MLIR**：僅針對 ONNX 格式設計，對其他模型格式的支持有限。因此，它適合使用 ONNX 格式模型的工作流程。
   - **TVM**：支持多種模型格式，包括 TensorFlow、PyTorch 和 MXNet，不僅限於 ONNX，適用於更多樣化的模型和框架。

3. **優化和調優能力**

   - **ONNX-MLIR**：主要依賴於 MLIR 和 LLVM 進行優化，著重於標準的計算圖優化，如常數折疊和算子融合。它更適合將 ONNX 模型編譯成高效的二進制文件。
   - **TVM**：具有強大的自動調優功能（Auto-Tuning），能夠根據硬件特性動態調整運行時參數，以獲得更好的性能。TVM 對於需要極致性能的場景更為適用，例如專門針對特定 GPU 架構的優化。

4. **生態系統和社群支持**

   - **ONNX-MLIR**：相對專注於 ONNX 格式，社群主要集中在 ONNX 和 MLIR 領域。
   - **TVM**：擁有更大的生態系統和開源社群，並受到各種硬件和雲服務提供商的支持（如 AWS、NVIDIA），且有多個企業和研究機構參與其中。

#### 適用場景

- **ONNX-MLIR** 更適合已經轉換為 ONNX 格式的模型部署工作流程，尤其是在架構比較固定的場景中使用，例如需要部署到多個 x86 或 ARM 的伺服器上。
- **TVM** 更適合需要多種硬件支持、不同模型格式，以及高度優化需求的場景。特別適合多樣化的深度學習應用場景，包括需要自動調優和跨架構部署的環境。

## 其他模型編譯和優化工具

除了 ONNX-MLIR 和 TVM，目前市面上還有其他幾個將機器學習模型編譯和優化為高效可執行文件的工具。這些工具各有特點，適合不同的模型格式、硬體架構和應用場景。以下是一些主流的模型編譯和優化工具：

| 編譯器工具              | 支持模型格式                   | 目標硬件                    | 主要應用場景            | 維護者                       |
| ------------------ | ------------------------ | ----------------------- | ----------------- | ------------------------- |
| **TensorRT**       | ONNX、TensorFlow、Caffe    | NVIDIA GPU              | 高性能 GPU 推理        | NVIDIA                    |
| **OpenVINO**       | ONNX、TensorFlow、Caffe    | Intel CPU、VPU、FPGA      | 邊緣推理、低功耗          | Intel                     |
| **XLA**            | TensorFlow、JAX           | TPU、GPU                 | 高效 TPU/GPU 訓練和推理  | Google                    |
| **Core ML**        | ONNX、TensorFlow、PyTorch  | Apple 設備                | 移動端、嵌入式           | Apple                     |
| **AWS Neuron**     | TensorFlow、PyTorch、MXNet | AWS Inferentia、Trainium | 雲端推理              | Amazon Web Services (AWS) |
| **Blade (OctoML)** | ONNX、TensorFlow、PyTorch  | 多硬體平台                   | 自動化優化、多硬體部署       | OctoML                    |
| **Glow**           | PyTorch、ONNX             | CPU、GPU、TPU             | PyTorch 集成、靈活硬件適配 | Meta (Facebook)           |
| **ONNX-MLIR**      | ONNX                     | 多種 CPU、部分 GPU 平台        | ONNX 模型優化與高效部署    | ONNX 社群                   |
| **TVM**            | ONNX、TensorFlow、PyTorch  | CPU、GPU、TPU、嵌入式設備       | 多平台自動優化和硬件特定優化    | Apache 基金會                |

## 總結

- **ONNX-MLIR** 是一個針對 ONNX 格式模型的專用編譯器，著重於直接且高效地將 ONNX 模型轉換為可執行文件，特別適合在架構相對固定的場景中使用。
- **TVM** 是更通用的深度學習編譯器，支持多種模型格式，並且在優化和硬件適配方面具有強大的靈活性，適合需要極致性能優化的多樣化深度學習應用。

## Reference
- [工欲利其器：深度學習編譯器介紹](https://ithelp.ithome.com.tw/articles/10209581?sc=rss.iron)


TVM安裝
https://tvm.apache.org/docs/install/from_source.html
https://tvm.hyper.ai/docs/install/from_source