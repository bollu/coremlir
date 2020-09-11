// Main
// Core2MLIR: GenMLIR BeforeCorePrep
module {
  // hask.make_data_constructor @"+#"
  // hask.make_data_constructor @"-#"
  // hask.make_data_constructor @"()"

  hask.func @fibstrict {
    %lambda = hask.lambdaSSA(%i: !hask.value) {
      %retval = hask.caseint  %i
      [@default -> { ^entry(%ival:!hask.value): // todo: remove this defult
        %fib_rec = hask.ref (@fibstrict):!hask.fn<!hask.value, !hask.value>
        %lit_one = hask.make_i64(1)
        %i_minus_one_v = hask.primop_sub(%ival, %lit_one)

        %fib_i_minus_one_t = hask.apSSA(%fib_rec: !hask.fn<!hask.value, !hask.value>, %i_minus_one_v)
        %fib_i_minus_one_v = hask.force(%fib_i_minus_one_t :!hask.value):!hask.value

        %lit_two = hask.make_i64(2)
        %i_minus_two_v = hask.primop_sub(%ival, %lit_two)
        %fib_i_minus_two_t = hask.apSSA(%fib_rec: !hask.fn<!hask.value, !hask.value>, %i_minus_two_v)
        %fib_i_minus_two_v = hask.force(%fib_i_minus_two_t : !hask.value): !hask.value

        %fib_sum  = hask.primop_add(%fib_i_minus_one_v, %fib_i_minus_two_v)                                                 

        hask.return(%fib_sum):!hask.value }]
      [0 -> { ^entry(%zero: !hask.value):
        hask.return(%zero):!hask.value }]
      [1 -> { ^entry(%one: !hask.value):
        hask.return(%one):!hask.value }]

      hask.return(%retval) : !hask.value
    }
    hask.return(%lambda) : !hask.fn<!hask.value, !hask.value>
  }

  // i:      0 1 2 3 4 5 6
  // fib(i): 0 1 1 2 3 5 8
  hask.func @main {
    %lambda = hask.lambdaSSA(%_: !hask.thunk) {
      %lit_6 = hask.make_i64(6)
      %fib = hask.ref(@fibstrict)  : !hask.fn<!hask.value, !hask.value>
      %out_v = hask.apSSA(%fib : !hask.fn<!hask.value, !hask.value>, %lit_6)
      %out_v_forced = hask.force(%out_v : !hask.value): !hask.value
      %x = hask.construct(@X, %out_v_forced)
      hask.return(%x) : !hask.thunk
    }
    hask.return(%lambda) :!hask.fn<!hask.thunk, !hask.thunk>
  }

}
