

module {
  hask.func @fibstrict {
    %0 = hask.lambdaSSA(%arg0:!hask.value) {
      %1 = hask.caseSSA %arg0 ["default" ->  {
        %2 = hask.ref(@fibstrict) : !hask.fn<!hask.value, !hask.value>
        %3 = hask.ref(@"-#") : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.value>>
        %4 = hask.make_i64(1 : i64)
        %5 = hask.apSSA(%3 :!hask.fn<!hask.value, !hask.fn<!hask.value, !hask.value>>, %arg0, %4)
        %6 = hask.apSSA(%2 :!hask.fn<!hask.value, !hask.value>, %5)
        %7 = hask.apSSA(%2 :!hask.fn<!hask.value, !hask.value>, %arg0)
        %8 = hask.ref(@"+#") : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.value>>
        %9 = hask.apSSA(%8 :!hask.fn<!hask.value, !hask.fn<!hask.value, !hask.value>>, %7, %6)
        hask.return(%9) : !hask.value
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
    hask.return(%0) : !hask.fn<!hask.value, !hask.value>
  }
}