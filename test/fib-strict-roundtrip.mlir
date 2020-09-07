

module {
  hask.func @fibstrict {
    %0 = hask.lambdaSSA(%arg0:!hask.value) {
      %1 = hask.caseSSA %arg0 ["default" ->  {
        %2 = hask.ref(@fibstrict)
        %3 = hask.ref(@"-#")
        %4 = hask.make_i64(1 : i64)
        %5 = hask.apSSA(%3, %arg0, %4)
        %6 = hask.apSSA(%2, %5)
        %7 = hask.ref(@"+#")
        %8 = hask.apSSA(%2, %arg0, %6)
        hask.return(%8) : !hask.value
      }]
 [0 : i64 ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        hask.return(%arg0) : !hask.value
      }]
 [1 : i64 ->  {
      ^bb0(%arg1: !hask.value):  // no predecessors
        hask.return(%arg0) : !hask.value
      }]

      hask.return(%1) : !hask.value
    }
    hask.return(%0) : !hask.value
  }
}