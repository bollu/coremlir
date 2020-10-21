

module {
  hask.func @k {
    %0 = hask.force(%arg0):!hask.value
    hask.return(%0) : !hask.value
  }
  hask.func @loop {
    %0 = hask.ref(@loop) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
    %1 = hask.ap(%0 :!hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>, %arg0)
    %2 = hask.force(%1):!hask.value
    hask.return(%2) : !hask.value
  }
  hask.adt @X [#hask.data_constructor<@MkX []>]
  hask.func @main {
    %0 = hask.make_i64(42 : i64)
    %1 = hask.construct(@X, %0 : !hask.value) : !hask.adt<@X>
    %2 = hask.transmute(%1 :!hask.adt<@X>):!hask.value
    %3 = hask.thunkify(%2 :!hask.value):!hask.thunk<!hask.value>
    %4 = hask.ref(@loop) : !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
    %5 = hask.ap(%4 :!hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>, %3)
    %6 = hask.ref(@k) : !hask.fn<(!hask.thunk<!hask.value>, !hask.thunk<!hask.value>) -> !hask.value>
    %7 = hask.ap(%6 :!hask.fn<(!hask.thunk<!hask.value>, !hask.thunk<!hask.value>) -> !hask.value>, %3, %5)
    %8 = hask.force(%7):!hask.value
    hask.return(%8) : !hask.value
  }
}