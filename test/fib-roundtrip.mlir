

module {
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]
  hask.func @plus {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk,%arg1:!hask.thunk) {
      %1 = hask.force(%arg0 :!hask.thunk):!hask.value
      %2 = hask.caseSSA %1 [@MkSimpleInt ->  {
      ^bb0(%arg2: !hask.value):  // no predecessors
        %3 = hask.force(%arg1 :!hask.thunk):!hask.value
        %4 = hask.caseSSA %3 [@MkSimpleInt ->  {
        ^bb0(%arg3: !hask.value):  // no predecessors
          %5 = hask.primop_add(%arg2,%arg3)
          %6 = hask.construct(@MkSimpleInt, %5)
          %7 = hask.thunkify(%6 :!hask.thunk):!hask.thunk
          hask.return(%7) : !hask.thunk
        }]

        hask.return(%4) : !hask.thunk
      }]

      hask.return(%2) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
  }
  hask.func @minus {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk,%arg1:!hask.thunk) {
      %1 = hask.force(%arg0 :!hask.thunk):!hask.value
      %2 = hask.caseSSA %1 [@MkSimpleInt ->  {
      ^bb0(%arg2: !hask.value):  // no predecessors
        %3 = hask.force(%arg1 :!hask.thunk):!hask.value
        %4 = hask.caseSSA %3 [@MkSimpleInt ->  {
        ^bb0(%arg3: !hask.value):  // no predecessors
          %5 = hask.primop_sub(%arg2,%arg3)
          %6 = hask.construct(@MkSimpleInt, %5)
          %7 = hask.thunkify(%6 :!hask.thunk):!hask.thunk
          hask.return(%7) : !hask.thunk
        }]

        hask.return(%4) : !hask.thunk
      }]

      hask.return(%2) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
  }
  hask.global @one {
    %0 = hask.make_i64(1 : i64)
    %1 = hask.construct(@MkSimpleInt, %0)
    hask.return(%1) : !hask.thunk
  }
  hask.global @zero {
    %0 = hask.make_i64(0 : i64)
    %1 = hask.construct(@MkSimpleInt, %0)
    hask.return(%1) : !hask.thunk
  }
  hask.global @two {
    %0 = hask.make_i64(2 : i64)
    %1 = hask.construct(@MkSimpleInt, %0)
    hask.return(%1) : !hask.thunk
  }
  hask.global @eight {
    %0 = hask.make_i64(8 : i64)
    %1 = hask.construct(@MkSimpleInt, %0)
    hask.return(%1) : !hask.thunk
  }
  hask.func @fib {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk) {
      %1 = hask.force(%arg0 :!hask.thunk):!hask.value
      %2 = hask.caseSSA %1 [@MkSimpleInt ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        %3 = hask.caseint %arg1 [0 : i64 ->  {
        ^bb0(%arg2: !hask.value):  // no predecessors
          %4 = hask.ref(@zero) : !hask.thunk
          %5 = hask.apSSA(%4 :!hask.thunk)
          hask.return(%5) : !hask.thunk
        }]
 [1 : i64 ->  {
        ^bb0(%arg2: !hask.value):  // no predecessors
          %4 = hask.ref(@one) : !hask.thunk
          %5 = hask.apSSA(%4 :!hask.thunk)
          hask.return(%5) : !hask.thunk
        }]
 [@default ->  {
          %4 = hask.ref(@eight) : !hask.thunk
          %5 = hask.ref(@fib) : !hask.fn<!hask.thunk, !hask.thunk>
          %6 = hask.ref(@minus) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
          %7 = hask.ref(@one) : !hask.thunk
          %8 = hask.apSSA(%6 :!hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>, %arg0, %7)
          %9 = hask.force(%8 :!hask.thunk):!hask.thunk
          %10 = hask.apSSA(%5 :!hask.fn<!hask.thunk, !hask.thunk>, %9)
          %11 = hask.force(%10 :!hask.thunk):!hask.thunk
          %12 = hask.ref(@two) : !hask.thunk
          %13 = hask.apSSA(%6 :!hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>, %arg0, %12)
          %14 = hask.force(%13 :!hask.thunk):!hask.thunk
          %15 = hask.force(%10 :!hask.thunk):!hask.thunk
          %16 = hask.ref(@plus) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
          %17 = hask.apSSA(%16 :!hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>, %11, %15)
          hask.return(%17) : !hask.thunk
        }]

        hask.return(%3) : !hask.thunk
      }]

      hask.return(%2) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.thunk>
  }
  hask.func @main {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk) {
      %1 = hask.ref(@one) : !hask.thunk
      %2 = hask.make_i64(8 : i64)
      %3 = hask.construct(@MkSimpleInt, %2)
      %4 = hask.thunkify(%3 :!hask.thunk):!hask.thunk
      %5 = hask.ref(@fib) : !hask.fn<!hask.thunk, !hask.thunk>
      %6 = hask.apSSA(%5 :!hask.fn<!hask.thunk, !hask.thunk>, %4)
      %7 = hask.force(%6 :!hask.thunk):!hask.thunk
      %8 = hask.force(%7 :!hask.thunk):!hask.thunk
      hask.return(%8) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.thunk>
  }
}