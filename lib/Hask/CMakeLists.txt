add_mlir_dialect_library(MLIRHask
        HaskDialect.cpp
        HaskOps.cpp
        HaskOptimization.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Hask

        DEPENDS
        MLIRHaskOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
