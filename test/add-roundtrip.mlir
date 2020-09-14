

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
          hask.return(%6) : !hask.thunk
        }]

        hask.return(%4) : !hask.thunk
      }]

      hask.return(%2) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
  }
  hask.global @zero {
    %0 = hask.make_i64(0 : i64)
    %1 = hask.construct(@MkSimpleInt, %0)
    hask.return(%1) : !hask.thunk
  }
  hask.global @one {
    %0 = hask.make_i64(1 : i64)
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
  hask.func @main {
    %0 = hask.lambdaSSA(%arg0:!hask.thunk) {
      %1 = hask.ref(@one) : !hask.thunk
      %2 = hask.apSSA(%1 :!hask.thunk)
      %3 = hask.ref(@two) : !hask.thunk
      %4 = hask.apSSA(%3 :!hask.thunk)
      %5 = hask.ref(@plus) : !hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>
      %6 = hask.apSSA(%5 :!hask.fn<!hask.thunk, !hask.fn<!hask.thunk, !hask.thunk>>, %2, %4)
      %7 = hask.force(%6 :!hask.thunk):!hask.thunk
      hask.return(%7) : !hask.thunk
    }
    hask.return(%0) : !hask.fn<!hask.thunk, !hask.thunk>
  }
}