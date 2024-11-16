

## 在 macOS 上使用 onnx-mlir 編譯一個共享庫（.so 文件）並讓它在 Linux x86_64 平台上運行
將生成的共享庫文件部署到嵌入式系統上，並使用適當的 C 或 C++ 程式碼加載該庫，然後進行推論。

安裝toolchains
https://github.com/messense/homebrew-macos-cross-toolchains

```sh
../onnx-mlir --EmitObj deploy_model.onnx --mtriple=x86_64-linux-gnu-g++
x86_64-unknown-linux-gnu-gcc -shared -o deploy_model.so deploy_model.o
```