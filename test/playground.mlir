module {
  hask.adt @X [#hask.data_constructor<@MkX []>]
  hask.func @main {
    %0 = hask.lambda(%arg0:!hask.thunk<!hask.value>) {
      %1 = hask.make_i64(42 : i64)
      %2 = hask.construct(@X, %1 : !hask.value) : !hask.adt<@X>
      %3 = hask.transmute(%2 :!hask.adt<@X>):!hask.value
      %4 = hask.thunkify(%3 :!hask.value):!hask.thunk<!hask.value>
      hask.return(%1) : !hask.value
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
  }
}
