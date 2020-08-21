

module {
  module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"
    func @fib(%arg0: i32) -> i32 {
      %c0_i32 = constant 0 : i32
      %0 = cmpi "eq", %c0_i32, %arg0 : i32
      cond_br %0, ^bb1(%arg0 : i32), ^bb2(%arg0 : i32)
    ^bb1(%1: i32):  // pred: ^bb0
      return %arg0 : i32
    ^bb2(%2: i32):  // pred: ^bb0
      %c1_i32 = constant 1 : i32
      %3 = cmpi "eq", %c1_i32, %arg0 : i32
      cond_br %3, ^bb3(%arg0 : i32), ^bb4(%arg0 : i32)
    ^bb3(%4: i32):  // pred: ^bb2
      return %arg0 : i32
    ^bb4(%5: i32):  // pred: ^bb2
      %c1_i32_0 = constant 1 : i32
      %6 = subi %arg0, %c1_i32_0 : i32
      %7 = call @fib(%6) : (i32) -> i32
      %8 = call @fib(%arg0) : (i32) -> i32
      %9 = addi %8, %7 : i32
      return %9 : i32
    }
  }
}