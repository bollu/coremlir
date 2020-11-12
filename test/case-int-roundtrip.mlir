

module {
  "hask.func"() ( {
  ^bb0(%arg0: !hask.value):  // no predecessors
    %0 = hask.caseint %arg0 [0 : i64 ->  {
    ^bb0(%arg1: !hask.value):  // no predecessors
      "hask.return"(%arg1) : (!hask.value) -> ()
    }]
 [@default ->  {
      %1 = "hask.make_i64"() {value = 1 : i64} : () -> !hask.value
      %2 = hask.primop_sub(%arg0,%1)
      "hask.return"(%2) : (!hask.value) -> ()
    }]

    "hask.return"(%0) : (!hask.value) -> ()
  }) {retty = !hask.value, sym_name = "prec"} : () -> ()
  "hask.func"() ( {
    %0 = "hask.make_i64"() {value = 42 : i64} : () -> !hask.value
    %1 = "hask.ref"() {sym_name = "prec"} : () -> !hask.fn<(!hask.value) -> !hask.value>
    %2 = "hask.ap"(%1, %0) : (!hask.fn<(!hask.value) -> !hask.value>, !hask.value) -> !hask.thunk<!hask.value>
    %3 = "hask.force"(%2) : (!hask.thunk<!hask.value>) -> !hask.value
    %4 = "hask.construct"(%3) {dataconstructor = @X} : (!hask.value) -> !hask.adt<@X>
    "hask.return"(%4) : (!hask.adt<@X>) -> ()
  }) {retty = !hask.value, sym_name = "main"} : () -> ()
}