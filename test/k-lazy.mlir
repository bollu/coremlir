// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s || true
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s || true
// CHECK: 42
module {
  // k x y = x
  hask.func @k (%x: !hask.thunk<!hask.value>, %y: !hask.thunk<!hask.value>) -> !hask.value {
      %x_v = hask.force(%x):!hask.value
      hask.return(%x_v) : !hask.value
    }

  // loop a = loop a
  hask.func @loop (%a: !hask.thunk<!hask.value>) -> !hask.value {
      %loop = hask.ref(@loop) : !hask.fn<(!hask.thunk<!hask.value>) ->  !hask.value>
      %out_t = hask.ap(%loop : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>, %a)
      %out_v = hask.force(%out_t) : !hask.value
      hask.return(%out_v) : !hask.value
    }

  hask.adt @X [#hask.data_constructor<@MkX []>]

  // k (x:(X 42)) (y:(loop (X 42))) = x
  // main = 
  //     let y = loop x -- builds a closure.
  //     in (k x y)
  hask.func @main ()  -> !hask.value {
      %lit_42 = hask.make_i64(42)
      // TODO: I need a think to transmute to different types.
      // Because we may want to "downcast" a ADT to a raw value
      %x = hask.construct(@X, %lit_42 : !hask.value): !hask.adt<@X>
      %x_v = hask.transmute(%x : !hask.adt<@X>): !hask.value
      %x_t = hask.thunkify(%x_v : !hask.value) :!hask.thunk<!hask.value>

      %loop = hask.ref(@loop) :  !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
      %y = hask.ap(%loop : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>, %x_t)


      %k = hask.ref(@k) : !hask.fn<(!hask.thunk<!hask.value>, !hask.thunk<!hask.value>) -> !hask.value>
      %out_t = hask.ap(%k: !hask.fn<(!hask.thunk<!hask.value>, !hask.thunk<!hask.value>) -> !hask.value>, 
        %x_t, %y)
      %out = hask.force(%out_t) : !hask.value
      hask.return(%out) : !hask.value
    }
}
