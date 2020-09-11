

module {
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]
  hask.func @plus {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk,%arg1:!hask.thunk) {
      %1 = hask.force(%arg0)
      %2 = hask.caseSSA %1 [@SimpleInt ->  {
      ^bb0(%arg2: !hask.value):  // no predecessors
        %3 = hask.force(%arg1)
        %4 = hask.caseSSA %3 [@SimpleInt ->  {
        ^bb0(%arg3: !hask.value):  // no predecessors
          %5 = hask.primop_add(%arg2,%arg3)
          %6 = hask.construct(@MkSimpleInt, %5)
          hask.return(%6) : !hask.thunk
        }]

        hask.return(%4) : !hask.thunk
      }]

      hask.return(%2) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
  }
  hask.func @minus {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk,%arg1:!hask.thunk) {
      %1 = hask.force(%arg0)
      %2 = hask.caseSSA %1 [@SimpleInt ->  {
      ^bb0(%arg2: !hask.value):  // no predecessors
        %3 = hask.force(%arg1)
        %4 = hask.caseSSA %3 [@SimpleInt ->  {
        ^bb0(%arg3: !hask.value):  // no predecessors
          %5 = hask.primop_sub(%arg2,%arg3)
          %6 = hask.construct(@MkSimpleInt, %5)
          hask.return(%6) : !hask.thunk
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
  hask.func @fib {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk) {
      %1 = hask.force(%arg0)
      %2 = hask.caseSSA %1 [@MkSimpleInt ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        %3 = hask.caseSSA %arg1 [0 : i64 ->  {
        ^bb0(%arg2: !hask.value):  // no predecessors
          %4 = hask.ref(@zero) : !hask.thunk
          hask.return(%4) : !hask.thunk
        }]
 [1 : i64 ->  {
        ^bb0(%arg2: !hask.value):  // no predecessors
          %4 = hask.ref(@one) : !hask.thunk
          hask.return(%4) : !hask.thunk
        }]
 [@default ->  {
          %4 = hask.ref(@fib) : !hask.fn<!hask.thunk, !hask.thunk>
          %5 = hask.apSSA(%4 :!hask.fn<!hask.thunk, !hask.thunk>, %arg0)
          %6 = hask.ref(@minus) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
          %7 = hask.ref(@one) : !hask.thunk
          %8 = hask.apSSA(%6 :!hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>, %arg0, %7)
          %9 = hask.apSSA(%4 :!hask.fn<!hask.thunk, !hask.thunk>, %8)
          %10 = hask.ref(@plus) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
          %11 = hask.apSSA(%10 :!hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>, %5, %9)
          hask.return(%11) : !hask.thunk
        }]

        hask.return(%3) : !hask.thunk
      }]

      hask.return(%2) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.thunk>
  }
}