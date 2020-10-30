module Core2MLIR.HaskMLIR where
import Core2MLIR.MLIR
import Outputable

fn :: ([Type], [Type]) -> Region -> Operation
fn (paramtys, retty) r = defaultop { opty = FunctionType paramtys retty, opregions = RegionList [r] }

-- make fib
ex1 :: SDoc
ex1 = error "foo"

