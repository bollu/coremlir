

module {
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]
  hask.func @plus {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk) {
      %1 = hask.lambdaSSA(%arg1:!hask.thunk) {
        %2 = hask.force(%arg0)
        %3 = hask.force(%2)
        %4 = hask.force(%arg1)
        %5 = hask.force(%4)
        %6 = hask.ref(@"+#")
        %7 = hask.apSSA(%6, %3, %5)
        %8 = hask.ref(@MkSimpleInt)
        %9 = hask.construct(@MkSimpleInt, %7)
        hask.return(%9) : !hask.thunk
      }
      hask.return(%1) : !hask.fn<!hask.thunk, !hask.thunk>
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
  }
  hask.func @minus {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk) {
      %1 = hask.lambdaSSA(%arg1:!hask.thunk) {
        %2 = hask.force(%arg0)
        %3 = hask.force(%2)
        %4 = hask.force(%arg1)
        %5 = hask.force(%4)
        %6 = hask.ref(@"-#")
        %7 = hask.apSSA(%6, %3, %5)
        %8 = hask.ref(@MkSimpleInt)
        %9 = hask.apSSA(%8, %7)
        hask.return(%9) : !hask.thunk
      }
      hask.return(%1) : !hask.fn<!hask.thunk, !hask.thunk>
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
  }
  hask.global @one {
    %0 = hask.ref(@MkSimpleInt)
    %1 = hask.make_i64(1 : i64)
    %2 = hask.apSSA(%0, %1)
    hask.return(%2) : !hask.thunk
  }
  hask.global @zero {
    %0 = hask.ref(@MkSimpleInt)
    %1 = hask.make_i64(0 : i64)
    %2 = hask.apSSA(%0, %1)
    hask.return(%2) : !hask.thunk
  }
  hask.func @fib {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk) {
      %1 = hask.force(%arg0)
      %2 = hask.force(%1)
      %3 = hask.caseSSA%2 [0 : i64 ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        %4 = hask.ref(@zero)
        hask.return(%4) : !hask.thunk
      }]
 [1 : i64 ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        %4 = hask.ref(@one)
        hask.return(%4) : !hask.thunk
      }]
 [@default ->  {
        %4 = hask.ref(@fib)
        %5 = hask.apSSA(%4, %arg0)
        %6 = hask.ref(@minus)
        %7 = hask.ref(@one)
        %8 = hask.apSSA(%6, %arg0, %7)
        %9 = hask.apSSA(%4, %8)
        %10 = hask.ref(@plus)
        %11 = hask.apSSA(%10, %5, %9)
        hask.return(%11) : !hask.thunk
      }]

      hask.return(%3) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.thunk>
  }
}