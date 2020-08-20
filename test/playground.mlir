// Debugging file: Can do anything here.
hask.module {
    %plus_hash = hask.make_data_constructor<"+#">
    %minus_hash = hask.make_data_constructor<"-#">
    %unit_tuple = hask.make_data_constructor<"()">
  hask.func @fib {
    %lambda = hask.lambdaSSA(%i) {
      %foo = hask.make_data_constructor<"foo">
      hask.return(%foo)
    }
    hask.return(%lambda)
  }
  hask.dummy_finish
}
