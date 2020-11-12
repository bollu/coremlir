

module {
  "hask.func"() ( {
  ^bb0(%arg0: !hask.value):  // no predecessors
    %0 = hask.caseint %arg0 [@default ->  {
      %1 = "hask.ref"() {sym_name = "fibstrict"} : () -> !hask.fn<(!hask.value) -> !hask.value>
      %2 = "hask.make_i64"() {value = 1 : i64} : () -> !hask.value
      %3 = hask.primop_sub(%arg0,%2)
      %4 = "hask.ap"(%1, %3) : (!hask.fn<(!hask.value) -> !hask.value>, !hask.value) -> !hask.thunk<!hask.value>
      %5 = "hask.force"(%4) : (!hask.thunk<!hask.value>) -> !hask.value
      %6 = "hask.make_i64"() {value = 2 : i64} : () -> !hask.value
      %7 = hask.primop_sub(%arg0,%6)
      %8 = "hask.ap"(%1, %7) : (!hask.fn<(!hask.value) -> !hask.value>, !hask.value) -> !hask.thunk<!hask.value>
      %9 = "hask.force"(%8) : (!hask.thunk<!hask.value>) -> !hask.value
      %10 = hask.primop_add(%5,%9)
      "hask.return"(%10) : (!hask.value) -> ()
    }]
 [0 : i64 ->  {
      "hask.return"(%arg0) : (!hask.value) -> ()
    }]
 [1 : i64 ->  {
      "hask.return"(%arg0) : (!hask.value) -> ()
    }]

    "hask.return"(%0) : (!hask.value) -> ()
  }) {retty = !hask.value, sym_name = "fibstrict"} : () -> ()
  "hask.func"() ( {
    %0 = "hask.make_i64"() {value = 6 : i64} : () -> !hask.value
    %1 = "hask.ref"() {sym_name = "fibstrict"} : () -> !hask.fn<(!hask.value) -> !hask.value>
    %2 = "hask.ap"(%1, %0) : (!hask.fn<(!hask.value) -> !hask.value>, !hask.value) -> !hask.thunk<!hask.value>
    %3 = "hask.force"(%2) : (!hask.thunk<!hask.value>) -> !hask.value
    %4 = "hask.construct"(%3) {dataconstructor = @X} : (!hask.value) -> !hask.adt<@X>
    "hask.return"(%4) : (!hask.adt<@X>) -> ()
  }) {retty = !hask.adt<@X>, sym_name = "main"} : () -> ()
}