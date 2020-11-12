

module {
  "hask.func"() ( {
  ^bb0(%arg0: !hask.thunk<!hask.value>, %arg1: !hask.thunk<!hask.value>):  // no predecessors
    %0 = "hask.force"(%arg0) : (!hask.thunk<!hask.value>) -> !hask.value
    "hask.return"(%0) : (!hask.value) -> ()
  }) {retty = !hask.value, sym_name = "k"} : () -> ()
  "hask.func"() ( {
  ^bb0(%arg0: !hask.thunk<!hask.value>):  // no predecessors
    %0 = "hask.ref"() {sym_name = "loop"} : () -> !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
    %1 = "hask.ap"(%0, %arg0) : (!hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>, !hask.thunk<!hask.value>) -> !hask.thunk<!hask.value>
    %2 = "hask.force"(%1) : (!hask.thunk<!hask.value>) -> !hask.value
    "hask.return"(%2) : (!hask.value) -> ()
  }) {retty = !hask.value, sym_name = "loop"} : () -> ()
  hask.adt @X [#hask.data_constructor<@MkX []>]
  "hask.func"() ( {
    %0 = "hask.make_i64"() {value = 42 : i64} : () -> !hask.value
    %1 = "hask.construct"(%0) {dataconstructor = @X} : (!hask.value) -> !hask.adt<@X>
    %2 = hask.transmute(%1 :!hask.adt<@X>):!hask.value
    %3 = hask.thunkify(%2 :!hask.value):!hask.thunk<!hask.value>
    %4 = "hask.ref"() {sym_name = "loop"} : () -> !hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>
    %5 = "hask.ap"(%4, %3) : (!hask.fn<(!hask.thunk<!hask.value>) -> !hask.value>, !hask.thunk<!hask.value>) -> !hask.thunk<!hask.value>
    %6 = "hask.ref"() {sym_name = "k"} : () -> !hask.fn<(!hask.thunk<!hask.value>, !hask.thunk<!hask.value>) -> !hask.value>
    %7 = "hask.ap"(%6, %3, %5) : (!hask.fn<(!hask.thunk<!hask.value>, !hask.thunk<!hask.value>) -> !hask.value>, !hask.thunk<!hask.value>, !hask.thunk<!hask.value>) -> !hask.thunk<!hask.value>
    %8 = "hask.force"(%7) : (!hask.thunk<!hask.value>) -> !hask.value
    "hask.return"(%8) : (!hask.value) -> ()
  }) {retty = !hask.value, sym_name = "main"} : () -> ()
}