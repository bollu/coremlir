

module {
  hask.func @fibstrict {
    %0 = hask.lambdaSSA(%arg0:!hask.value) {
      %1 = hask.caseSSA %arg0 ["default" ->  {
        %2 = hask.ref(@fibstrict) : !hask.fn<!hask.value, !hask.value>
        %3 = hask.ref(@"-#") : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk>>
        %4 = hask.make_i64(1 : i64)
        %5 = hask.apSSA(%3 :!hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk>>, %arg0, %4)
        %6 = hask.force(%5)
        %7 = hask.apSSA(%2 :!hask.fn<!hask.value, !hask.value>, %6)
        %8 = hask.apSSA(%2 :!hask.fn<!hask.value, !hask.value>, %arg0)
        %9 = hask.ref(@"+#") : !hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk>>
        %10 = hask.apSSA(%9 :!hask.fn<!hask.value, !hask.fn<!hask.value, !hask.thunk>>, %8, %7)
        %11 = hask.force(%10)
        hask.return(%11) : !hask.value
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