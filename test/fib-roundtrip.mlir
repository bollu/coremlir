

module {
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]
  hask.func @plus {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.untyped>,%arg1:!hask.thunk<!hask.untyped>) {
      %1 = hask.force(%arg0 :!hask.thunk<!hask.untyped>):!hask.value
      %2 = hask.caseSSA %1 [@MkSimpleInt ->  {
      ^bb0(%arg2: !hask.value):  // no predecessors
        %3 = hask.force(%arg1 :!hask.thunk<!hask.untyped>):!hask.value
        %4 = hask.caseSSA %3 [@MkSimpleInt ->  {
        ^bb0(%arg3: !hask.value):  // no predecessors
          %5 = hask.primop_add(%arg2,%arg3)
          %6 = hask.construct(@MkSimpleInt, %5)
          hask.return(%6) : !hask.thunk<!hask.untyped>
        }]

        hask.return(%4) : !hask.thunk<!hask.untyped>
      }]

      hask.return(%2) : !hask.thunk<!hask.untyped>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
  }
  hask.func @minus {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.untyped>,%arg1:!hask.thunk<!hask.untyped>) {
      %1 = hask.force(%arg0 :!hask.thunk<!hask.untyped>):!hask.value
      %2 = hask.caseSSA %1 [@MkSimpleInt ->  {
      ^bb0(%arg2: !hask.value):  // no predecessors
        %3 = hask.force(%arg1 :!hask.thunk<!hask.untyped>):!hask.value
        %4 = hask.caseSSA %3 [@MkSimpleInt ->  {
        ^bb0(%arg3: !hask.value):  // no predecessors
          %5 = hask.primop_sub(%arg2,%arg3)
          %6 = hask.construct(@MkSimpleInt, %5)
          hask.return(%6) : !hask.thunk<!hask.untyped>
        }]

        hask.return(%4) : !hask.thunk<!hask.untyped>
      }]

      hask.return(%2) : !hask.thunk<!hask.untyped>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
  }
  hask.global @one {
    %0 = hask.make_i64(1 : i64)
    %1 = hask.construct(@MkSimpleInt, %0)
    hask.return(%1) : !hask.thunk<!hask.untyped>
  }
  hask.global @zero {
    %0 = hask.make_i64(0 : i64)
    %1 = hask.construct(@MkSimpleInt, %0)
    hask.return(%1) : !hask.thunk<!hask.untyped>
  }
  hask.global @two {
    %0 = hask.make_i64(2 : i64)
    %1 = hask.construct(@MkSimpleInt, %0)
    hask.return(%1) : !hask.thunk<!hask.untyped>
  }
  hask.global @eight {
    %0 = hask.make_i64(8 : i64)
    %1 = hask.construct(@MkSimpleInt, %0)
    hask.return(%1) : !hask.thunk<!hask.untyped>
  }
  hask.func @fib {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.untyped>) {
      %1 = hask.force(%arg0 :!hask.thunk<!hask.untyped>):!hask.value
      %2 = hask.caseSSA %1 [@MkSimpleInt ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        %3 = hask.caseint %arg1 [0 : i64 ->  {
        ^bb0(%arg2: !hask.value):  // no predecessors
          %4 = hask.ref(@zero) : !hask.thunk<!hask.untyped>
          %5 = hask.apSSA(%4 :!hask.thunk<!hask.untyped>)
          %6 = hask.force(%5 :!hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
          hask.return(%6) : !hask.thunk<!hask.untyped>
        }]
 [1 : i64 ->  {
        ^bb0(%arg2: !hask.value):  // no predecessors
          %4 = hask.ref(@one) : !hask.thunk<!hask.untyped>
          %5 = hask.apSSA(%4 :!hask.thunk<!hask.untyped>)
          %6 = hask.force(%5 :!hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
          hask.return(%6) : !hask.thunk<!hask.untyped>
        }]
 [@default ->  {
          %4 = hask.ref(@fib) : !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
          %5 = hask.ref(@minus) : !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
          %6 = hask.ref(@one) : !hask.thunk<!hask.untyped>
          %7 = hask.apSSA(%6 :!hask.thunk<!hask.untyped>)
          %8 = hask.apSSA(%5 :!hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, %arg0, %7)
          %9 = hask.apSSA(%4 :!hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, %8)
          %10 = hask.force(%9 :!hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
          %11 = hask.ref(@two) : !hask.thunk<!hask.untyped>
          %12 = hask.apSSA(%11 :!hask.thunk<!hask.untyped>)
          %13 = hask.apSSA(%5 :!hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, %arg0, %12)
          %14 = hask.apSSA(%4 :!hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, %13)
          %15 = hask.force(%14 :!hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
          %16 = hask.thunkify(%10 :!hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
          %17 = hask.thunkify(%15 :!hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
          %18 = hask.ref(@plus) : !hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
          %19 = hask.apSSA(%18 :!hask.fn<(!hask.thunk<!hask.untyped>, !hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, %16, %17)
          %20 = hask.force(%19 :!hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
          hask.return(%20) : !hask.thunk<!hask.untyped>
        }]

        hask.return(%3) : !hask.thunk<!hask.untyped>
      }]

      hask.return(%2) : !hask.thunk<!hask.untyped>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
  }
  hask.func @main {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk<!hask.untyped>) {
      %1 = hask.make_i64(6 : i64)
      %2 = hask.construct(@MkSimpleInt, %1)
      %3 = hask.thunkify(%2 :!hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
      %4 = hask.ref(@fib) : !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
      %5 = hask.apSSA(%4 :!hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>, %3)
      %6 = hask.force(%5 :!hask.thunk<!hask.untyped>):!hask.thunk<!hask.untyped>
      hask.return(%6) : !hask.thunk<!hask.untyped>
    }
    hask.return(%0) : !hask.fn<(!hask.thunk<!hask.untyped>) -> !hask.thunk<!hask.untyped>>
  }
}