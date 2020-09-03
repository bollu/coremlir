

module {
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]
  hask.func @plus {
    %0 = hask.lambdaSSA(%arg0) {
      %1 = hask.lambdaSSA(%arg1) {
        %2 = hask.force(%arg0)
        %3 = hask.force(%arg1)
        %4 = hask.ref(@"+#")
        %5 = hask.apSSA(%4, %2, %3)
        %6 = hask.ref(@MkSimpleInt)
        %7 = hask.apSSA(%6, %5)
        hask.return(%7)
      }
      hask.return(%1)
    }
    hask.return(%0)
  }
  hask.func @minus {
    %0 = hask.lambdaSSA(%arg0) {
      %1 = hask.lambdaSSA(%arg1) {
        %2 = hask.force(%arg0)
        %3 = hask.force(%arg1)
        %4 = hask.ref(@"-#")
        %5 = hask.apSSA(%4, %2, %3)
        %6 = hask.ref(@MkSimpleInt)
        %7 = hask.apSSA(%6, %5)
        hask.return(%7)
      }
      hask.return(%1)
    }
    hask.return(%0)
  }
  hask.global @one {
    %0 = hask.ref(@MkSimpleInt)
    %1 = hask.make_i64(1 : i64)
    %2 = hask.apSSA(%0, %1)
    hask.return(%2)
  }
  hask.global @zero {
    %0 = hask.ref(@MkSimpleInt)
    %1 = hask.make_i64(0 : i64)
    %2 = hask.apSSA(%0, %1)
    hask.return(%2)
  }
  hask.func @fib {
    %0 = hask.lambdaSSA(%arg0) {
      %1 = hask.force(%arg0)
      %2 = hask.caseSSA %1 [0 : i64 ->  {
      ^bb0(%arg1: !hask.untyped):  // no predecessors
        %3 = hask.ref(@zero)
        hask.return(%3)
      }]
 [1 : i64 ->  {
      ^bb0(%arg1: !hask.untyped):  // no predecessors
        %3 = hask.ref(@one)
        hask.return(%3)
      }]
 [@default ->  {
        %3 = hask.ref(@fib)
        %4 = hask.ref(@minus)
        %5 = hask.ref(@one)
        %6 = hask.apSSA(%4, %arg0, %5)
        %7 = hask.apSSA(%3, %6)
        %8 = hask.ref(@plus)
        %9 = hask.apSSA(%8, %7)
        hask.return(%9)
      }]

      hask.return(%2)
    }
    hask.return(%0)
  }
}