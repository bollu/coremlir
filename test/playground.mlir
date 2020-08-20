// Debugging file: Can do anything here.
hask.module {
    %plus_hash = hask.make_data_constructor<"+#">
    %minus_hash = hask.make_data_constructor<"-#">
    %unit_tuple = hask.make_data_constructor<"()">
  hask.func @fib {
    %lambda = hask.lambdaSSA(%i) {
      hask.return(%unit_tuple)
    }
    hask.return(%lambda)
  }
  hask.dummy_finish
}
