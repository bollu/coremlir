// Debugging file: Can do anything here.
hask.module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"

  hask.func @fib {
    %lambda = hask.lambdaSSA(%i) {
      // %foo_ref = constant @XXXX : () -> ()
      %f = hask.ref(@"+#")
      hask.return(%f)
    }
    hask.return(%lambda)
  }
  hask.dummy_finish
}
