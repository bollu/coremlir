

module {
  hask.adt @EitherBox [#hask.data_constructor<@Left [@Box]>, #hask.data_constructor<@Right [@Box]>]
  hask.adt @SimpleInt [#hask.data_constructor<@MkSimpleInt [@"Int#"]>]
  "hask.func"() ( {
  ^bb0(%arg0: !hask.thunk<!hask.adt<@Either>>):  // no predecessors
    %0 = "hask.force"(%arg0) : (!hask.thunk<!hask.adt<@Either>>) -> !hask.adt<@Either>
    %1 = "hask.case"(%0) ( {
    ^bb0(%arg1: !hask.thunk<!hask.adt<@Either>>):  // no predecessors
      %2 = "hask.force"(%arg1) : (!hask.thunk<!hask.adt<@Either>>) -> !hask.adt<@Either>
      %3 = "hask.case"(%2) ( {
      ^bb0(%arg2: !hask.thunk<!hask.value>):  // no predecessors
        %4 = "hask.force"(%arg2) : (!hask.thunk<!hask.value>) -> !hask.value
        "hask.return"(%4) : (!hask.value) -> ()
      }) {alt0 = @Left, constructorName = @Either} : (!hask.adt<@Either>) -> !hask.value
      "hask.return"(%3) : (!hask.value) -> ()
    }) {alt0 = @Right, constructorName = @Either} : (!hask.adt<@Either>) -> !hask.value
    "hask.return"(%1) : (!hask.value) -> ()
  }) {retty = !hask.value, sym_name = "extract"} : () -> ()
  "hask.func"() ( {
    %0 = "hask.make_i64"() {value = 1 : i64} : () -> !hask.value
    %1 = "hask.construct"(%0) {dataconstructor = @SimpleInt} : (!hask.value) -> !hask.adt<@SimpleInt>
    "hask.return"(%1) : (!hask.adt<@SimpleInt>) -> ()
  }) {retty = !hask.adt<@SimpleInt>, sym_name = "one"} : () -> ()
  "hask.func"() ( {
    %0 = "hask.ref"() {sym_name = "one"} : () -> !hask.fn<() -> !hask.adt<@SimpleInt>>
    %1 = "hask.ap"(%0) : (!hask.fn<() -> !hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
    %2 = "hask.construct"(%1) {dataconstructor = @Left} : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@Either>
    "hask.return"(%2) : (!hask.adt<@Either>) -> ()
  }) {retty = !hask.adt<@Either>, sym_name = "leftOne"} : () -> ()
  "hask.func"() ( {
    %0 = "hask.ref"() {sym_name = "leftOne"} : () -> !hask.fn<() -> !hask.adt<@Either>>
    %1 = "hask.ap"(%0) : (!hask.fn<() -> !hask.adt<@Either>>) -> !hask.thunk<!hask.adt<@Either>>
    %2 = "hask.construct"(%1) {dataconstructor = @Right} : (!hask.thunk<!hask.adt<@Either>>) -> !hask.adt<@Either>
    "hask.return"(%2) : (!hask.adt<@Either>) -> ()
  }) {retty = !hask.adt<@Either>, sym_name = "rightLeftOne"} : () -> ()
  "hask.func"() ( {
    %0 = "hask.ref"() {sym_name = "rightLeftOne"} : () -> !hask.fn<() -> !hask.adt<@Either>>
    %1 = "hask.ap"(%0) : (!hask.fn<() -> !hask.adt<@Either>>) -> !hask.thunk<!hask.adt<@Either>>
    %2 = "hask.ref"() {sym_name = "extract"} : () -> !hask.fn<(!hask.thunk<!hask.adt<@Either>>) -> !hask.value>
    %3 = "hask.ap"(%2, %1) : (!hask.fn<(!hask.thunk<!hask.adt<@Either>>) -> !hask.value>, !hask.thunk<!hask.adt<@Either>>) -> !hask.thunk<!hask.value>
    %4 = "hask.force"(%3) : (!hask.thunk<!hask.value>) -> !hask.value
    "hask.return"(%4) : (!hask.value) -> ()
  }) {retty = !hask.value, sym_name = "main"} : () -> ()
}