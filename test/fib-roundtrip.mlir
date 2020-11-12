

module {
  hask.adt @SimpleInt [#hask.data_constructor<@SimpleInt [@"Int#"]>]
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
  ^bb0(%arg0: !hask.thunk<!hask.adt<@SimpleInt>>, %arg1: !hask.thunk<!hask.adt<@SimpleInt>>):  // no predecessors
    %0 = "hask.force"(%arg0) : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>
    %1 = "hask.case"(%0) ( {
    ^bb0(%arg2: !hask.value):  // no predecessors
      %2 = "hask.force"(%arg1) : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>
      %3 = "hask.case"(%2) ( {
      ^bb0(%arg3: !hask.value):  // no predecessors
        %4 = hask.primop_sub(%arg2,%arg3)
        %5 = "hask.construct"(%4) {dataconstructor = @SimpleInt} : (!hask.value) -> !hask.adt<@SimpleInt>
        "hask.return"(%5) : (!hask.adt<@SimpleInt>) -> ()
      }) {alt0 = @SimpleInt, constructorName = @SimpleInt} : (!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>
      "hask.return"(%3) : (!hask.adt<@SimpleInt>) -> ()
    }) {alt0 = @SimpleInt, constructorName = @SimpleInt} : (!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>
    "hask.return"(%1) : (!hask.adt<@SimpleInt>) -> ()
  }) {retty = !hask.adt<@SimpleInt>, sym_name = "minus"} : () -> ()
  "hask.func"() ( {
    %0 = "hask.make_i64"() {value = 0 : i64} : () -> !hask.value
    %1 = "hask.construct"(%0) {dataconstructor = @SimpleInt} : (!hask.value) -> !hask.adt<@SimpleInt>
    "hask.return"(%1) : (!hask.adt<@SimpleInt>) -> ()
  }) {retty = !hask.adt<@SimpleInt>, sym_name = "zero"} : () -> ()
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
    %0 = "hask.make_i64"() {value = 8 : i64} : () -> !hask.value
    %1 = "hask.construct"(%0) {dataconstructor = @SimpleInt} : (!hask.value) -> !hask.adt<@SimpleInt>
    "hask.return"(%1) : (!hask.adt<@SimpleInt>) -> ()
  }) {retty = !hask.adt<@SimpleInt>, sym_name = "eight"} : () -> ()
  "hask.func"() ( {
  ^bb0(%arg0: !hask.thunk<!hask.adt<@SimpleInt>>):  // no predecessors
    %0 = "hask.force"(%arg0) : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>
    %1 = "hask.case"(%0) ( {
    ^bb0(%arg1: !hask.value):  // no predecessors
      %2 = hask.caseint %arg1 [0 : i64 ->  {
        %3 = "hask.ref"() {sym_name = "zero"} : () -> !hask.fn<() -> !hask.adt<@SimpleInt>>
        %4 = "hask.ap"(%3) : (!hask.fn<() -> !hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
        %5 = "hask.force"(%4) : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>
        "hask.return"(%5) : (!hask.adt<@SimpleInt>) -> ()
      }]
 [1 : i64 ->  {
        %3 = "hask.ref"() {sym_name = "one"} : () -> !hask.fn<() -> !hask.adt<@SimpleInt>>
        %4 = "hask.ap"(%3) : (!hask.fn<() -> !hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
        %5 = "hask.force"(%4) : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>
        "hask.return"(%5) : (!hask.adt<@SimpleInt>) -> ()
      }]
 [@default ->  {
        %3 = "hask.ref"() {sym_name = "fib"} : () -> !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
        %4 = "hask.ref"() {sym_name = "minus"} : () -> !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
        %5 = "hask.ref"() {sym_name = "one"} : () -> !hask.fn<() -> !hask.adt<@SimpleInt>>
        %6 = "hask.ap"(%5) : (!hask.fn<() -> !hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
        %7 = "hask.ap"(%4, %arg0, %6) : (!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
        %8 = "hask.ap"(%3, %7) : (!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
        %9 = "hask.force"(%8) : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>
        %10 = "hask.ref"() {sym_name = "two"} : () -> !hask.fn<() -> !hask.adt<@SimpleInt>>
        %11 = "hask.ap"(%10) : (!hask.fn<() -> !hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
        %12 = "hask.ap"(%4, %arg0, %11) : (!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
        %13 = "hask.ap"(%3, %12) : (!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
        %14 = "hask.force"(%13) : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>
        %15 = hask.thunkify(%9 :!hask.adt<@SimpleInt>):!hask.thunk<!hask.adt<@SimpleInt>>
        %16 = hask.thunkify(%14 :!hask.adt<@SimpleInt>):!hask.thunk<!hask.adt<@SimpleInt>>
        %17 = "hask.ref"() {sym_name = "plus"} : () -> !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
        %18 = "hask.ap"(%17, %15, %16) : (!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
        %19 = "hask.force"(%18) : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>
        "hask.return"(%19) : (!hask.adt<@SimpleInt>) -> ()
      }]

      "hask.return"(%2) : (!hask.adt<@SimpleInt>) -> ()
    }) {alt0 = @SimpleInt, constructorName = @SimpleInt} : (!hask.adt<@SimpleInt>) -> !hask.adt<@SimpleInt>
    "hask.return"(%1) : (!hask.adt<@SimpleInt>) -> ()
  }) {retty = !hask.adt<@SimpleInt>, sym_name = "fib"} : () -> ()
  "hask.func"() ( {
    %0 = "hask.make_i64"() {value = 6 : i64} : () -> !hask.value
    %1 = "hask.construct"(%0) {dataconstructor = @SimpleInt} : (!hask.value) -> !hask.adt<@SimpleInt>
    %2 = hask.thunkify(%1 :!hask.adt<@SimpleInt>):!hask.thunk<!hask.adt<@SimpleInt>>
    %3 = "hask.ref"() {sym_name = "fib"} : () -> !hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>
    %4 = "hask.ap"(%3, %2) : (!hask.fn<(!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>>, !hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.thunk<!hask.adt<@SimpleInt>>
    %5 = "hask.force"(%4) : (!hask.thunk<!hask.adt<@SimpleInt>>) -> !hask.adt<@SimpleInt>
    "hask.return"(%5) : (!hask.adt<@SimpleInt>) -> ()
  }) {retty = !hask.adt<@SimpleInt>, sym_name = "main"} : () -> ()
}