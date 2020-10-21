

module {
  hask.func @fibstrict {
    %0 = hask.caseint %arg0 [@default ->  {
      %1 = hask.ref(@fibstrict) : !hask.fn<(!hask.value) -> !hask.value>
      %2 = hask.make_i64(1 : i64)
      %3 = hask.primop_sub(%arg0,%2)
      %4 = hask.ap(%1 :!hask.fn<(!hask.value) -> !hask.value>, %3)
      %5 = hask.force(%4):!hask.value
      %6 = hask.make_i64(2 : i64)
      %7 = hask.primop_sub(%arg0,%6)
      %8 = hask.ap(%1 :!hask.fn<(!hask.value) -> !hask.value>, %7)
      %9 = hask.force(%8):!hask.value
      %10 = hask.primop_add(%5,%9)
      hask.return(%10) : !hask.value
    }]
 [0 : i64 ->  {
      hask.return(%arg0) : !hask.value
    }]
 [1 : i64 ->  {
      hask.return(%arg0) : !hask.value
    }]

    hask.return(%0) : !hask.value
  }
  hask.func @main {
    %0 = hask.make_i64(6 : i64)
    %1 = hask.ref(@fibstrict) : !hask.fn<(!hask.value) -> !hask.value>
    %2 = hask.ap(%1 :!hask.fn<(!hask.value) -> !hask.value>, %0)
    %3 = hask.force(%2):!hask.value
    %4 = hask.construct(@X, %3 : !hask.value) : !hask.adt<@X>
    hask.return(%4) : !hask.adt<@X>
  }
}