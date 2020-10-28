// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -interpret |  FileCheck %s
// CHECK: constructor(X 8)
module {
  // hask.make_data_constructor @"+#"
  // hask.make_data_constructor @"-#"
  // hask.make_data_constructor @"()"

  hask.func @fibstrict (%i: !hask.value) -> !hask.value {
      %retval = hask.caseint  %i
      [@default -> { ^entry: // todo: remove this defult
        %fib_rec = hask.ref (@fibstrict):!hask.fn<(!hask.value) -> !hask.value>
        %lit_one = hask.make_i64(1)
        %i_minus_one_v = hask.primop_sub(%i, %lit_one)

        %fib_i_minus_one_t = hask.ap(%fib_rec: !hask.fn<(!hask.value)->  !hask.value>, %i_minus_one_v)
        %fib_i_minus_one_v = hask.force(%fib_i_minus_one_t):!hask.value

        %lit_two = hask.make_i64(2)
        %i_minus_two_v = hask.primop_sub(%i, %lit_two)
        %fib_i_minus_two_t = hask.ap(%fib_rec: !hask.fn<(!hask.value) -> !hask.value>, %i_minus_two_v)
        %fib_i_minus_two_v = hask.force(%fib_i_minus_two_t): !hask.value

        %fib_sum  = hask.primop_add(%fib_i_minus_one_v, %fib_i_minus_two_v)                                                 
        hask.return(%fib_sum):!hask.value }]
      [0 -> { ^entry:
        hask.return(%i):!hask.value }]
      [1 -> { ^entry:
        hask.return(%i):!hask.value }]

      hask.return(%retval) : !hask.value
    }

  // i:      0 1 2 3 4 5 6
  // fib(i): 0 1 1 2 3 5 8
  hask.func @main () -> !hask.adt<@X> {
      %lit_6 = hask.make_i64(6)
      %fib = hask.ref(@fibstrict)  : !hask.fn<(!hask.value) -> !hask.value>
      %out_v = hask.ap(%fib : !hask.fn<(!hask.value) -> !hask.value>, %lit_6)
      %out_v_forced = hask.force(%out_v): !hask.value
      %x = hask.construct(@X, %out_v_forced: !hask.value): !hask.adt<@X>
      hask.return(%x) : !hask.adt<@X>
    }

}
