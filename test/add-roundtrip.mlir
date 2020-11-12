

module {
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]
  "hask.func"() ( {
  ^bb0(%arg0: !hask.thunk<!hask.adt<@SimpleInt>>, %arg1: !hask.thunk<!hask.adt<@SimpleInt>>):  // no predecessors
    %0 = "hask.force"(%arg0) : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>
    %1 = "hask.case"(%0) ( {
    ^bb0(%arg2: !hask.value):  // no predecessors
      %2 = "hask.force"(%arg1) : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>
      %3 = "hask.case"(%2) ( {
      ^bb0(%arg3: !hask.value):  // no predecessors
        %4 = hask.primop_add(%arg2,%arg3)
        %5 = "hask.construct"(%4) {dataconstructor = @SimpleInt} : (!hask.value) -> !hask.adt<@SimpleInt>
        "hask.return"(%5) : (!hask.adt<@SimpleInt>) -> ()
      }) {alt0 = @SimpleInt, constructorName = @SimpleInt} : (!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>
      "hask.return"(%3) : (!hask.adt<@SimpleInt>) -> ()
    }) {alt0 = @SimpleInt, constructorName = @SimpleInt} : (!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>
    "hask.return"(%1) : (!hask.adt<@SimpleInt>) -> ()
  }) {retty = !hask.adt<@SimpleInt>, sym_name = "plus"} : () -> ()
  "hask.func"() ( {
    %0 = "hask.make_i64"() {value = 1 : i64} : () -> !hask.value
    %1 = "hask.construct"(%0) {dataconstructor = @SimpleInt} : (!hask.value) -> !hask.adt<@SimpleInt>
    "hask.return"(%1) : (!hask.adt<@SimpleInt>) -> ()
  }) {retty = !hask.adt<@SimpleInt>, sym_name = "one"} : () -> ()
  "hask.func"() ( {
    %0 = "hask.make_i64"() {value = 2 : i64} : () -> !hask.value
    %1 = "hask.construct"(%0) {dataconstructor = @SimpleInt} : (!hask.value) -> !hask.adt<@SimpleInt>
    "hask.return"(%1) : (!hask.adt<@SimpleInt>) -> ()
  }) {retty = !hask.adt<@SimpleInt>, sym_name = "two"} : () -> ()
  "hask.func"() ( {
    %0 = "hask.ref"() {sym_name = "one"} : () -> !hask.fn<() -> !hask.adt<@SimpleInt>>
    %1 = "hask.ap"(%0) : (!hask.fn<() -> !hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
    %2 = "hask.ref"() {sym_name = "two"} : () -> !hask.fn<() -> !hask.adt<@SimpleInt>>
    %3 = "hask.ap"(%2) : (!hask.fn<() -> !hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
    %4 = "hask.ref"() {sym_name = "plus"} : () -> !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
    %5 = "hask.ap"(%4, %1, %3) : (!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
    %6 = "hask.force"(%5) : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>
    "hask.return"(%6) : (!hask.adt<@SimpleInt>) -> ()
  }) {retty = !hask.adt<@SimpleInt>, sym_name = "main"} : () -> ()
}