

module {
  module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"
    func @fib(%arg0: i64) -> i64 {
      %c0_i64 = constant 0 : i64
      %0 = cmpi "eq", %c0_i64, %arg0 : i64
      cond_br %0, ^bb1(%arg0 : i64), ^bb2(%arg0 : i64)
    ^bb1(%1: i64):  // pred: ^bb0
      return %arg0 : i64
    ^bb2(%2: i64):  // pred: ^bb0
      %c1_i64 = constant 1 : i64
      %3 = cmpi "eq", %c1_i64, %arg0 : i64
      cond_br %3, ^bb3(%arg0 : i64), ^bb4(%arg0 : i64)
    ^bb3(%4: i64):  // pred: ^bb2
      return %arg0 : i64
    ^bb4(%5: i64):  // pred: ^bb2
      %c1_i64_0 = constant 1 : i64
      %6 = subi %arg0, %c1_i64_0 : i64
      %7 = call @fib(%6) : (i64) -> i64
      %8 = call @fib(%arg0) : (i64) -> i64
      %9 = addi %8, %7 : i64
      return %9 : i64
    }
  }
}