

module {
  hask.func @fibstrict {
    %0 = hask.lambda(%arg0:!hask.value) {
      %1 = hask.caseint %arg0 [@default ->  {
        %2 = hask.ref(@fibstrict) : !hask.fn<(!hask.value) -> !hask.value>
        %3 = hask.make_i64(1 : i64)
        %4 = hask.primop_sub(%arg0,%3)
        %5 = hask.ap(%2 :!hask.fn<(!hask.value) -> !hask.value>, %4)
        %6 = hask.force(%5):!hask.value
        %7 = hask.make_i64(2 : i64)
        %8 = hask.primop_sub(%arg0,%7)
        %9 = hask.ap(%2 :!hask.fn<(!hask.value) -> !hask.value>, %8)
        %10 = hask.force(%9):!hask.value
        %11 = hask.primop_add(%6,%10)
        hask.return(%11) : !hask.value
      }]
 [0 : i64 ->  {
        hask.return(%arg0) : !hask.value
      }]
 [1 : i64 ->  {
        hask.return(%arg0) : !hask.value
      }]

      hask.return(%1) : !hask.value
    }
    hask.return(%0) : !hask.fn<(!hask.value) -> !hask.value>
  }
  hask.func @main {
    %0 = hask.lambda() {
      %1 = hask.make_i64(6 : i64)
      %2 = hask.ref(@fibstrict) : !hask.fn<(!hask.value) -> !hask.value>
      %3 = hask.ap(%2 :!hask.fn<(!hask.value) -> !hask.value>, %1)
      %4 = hask.force(%3):!hask.value
      %5 = hask.construct(@X, %4 : !hask.value) : !hask.adt<@X>
      hask.return(%5) : !hask.adt<@X>
    }
    hask.return(%0) : !hask.fn<() -> !hask.adt<@X>>
  }
}