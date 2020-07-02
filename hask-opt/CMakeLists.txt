get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRHask
        )
add_llvm_executable(hask-opt hask-opt.cpp)

llvm_update_compile_flags(hask-opt)
target_link_libraries(hask-opt PRIVATE ${LIBS})
