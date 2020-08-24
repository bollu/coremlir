# LLVM components
# https://github.com/llvm/llvm-project/blob/a93514abf2d843339878d0607e2f5fd2a787bf72/mlir/examples/toy/Ch7/CMakeLists.txt
set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_llvm_executable(hask-opt hask-opt.cpp)

llvm_update_compile_flags(hask-opt)
target_link_libraries(hask-opt PRIVATE
        ${dialect_libs}
        ${conversion_libs}
        MLIRTranslation
        MLIROptLib
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIR
        MLIRTransforms
        MLIRSupport)
mlir_check_link_libraries(hask-opt)
