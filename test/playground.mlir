// Debugging file: Can do anything here.
hask.module {
    hask.make_data_constructor @"+#"
    hask.make_data_constructor @"-#"
    hask.make_data_constructor @"()"

  hask.func @fib {
    %lambda = hask.lambdaSSA(%i) {
      %retval = hask.caseSSA  %i
      ["default" -> { ^entry(%default_random_name: !hask.untyped): // todo: remove this defult
        %i_minus = hask.apSSA(@"-#", %i)
        %lit_one = hask.make_i32(1)
        %i_minus_one = hask.apSSA(%i_minus, %lit_one)
        %fib_i_minus_one = hask.apSSA(@fib, %i_minus_one)
        %force_fib_i_minus_one = hask.force (%fib_i_minus_one) // todo: this is extraneous!
        %fib_i = hask.apSSA(@fib, %i)
        %force_fib_i = hask.force (%fib_i) // todo: this is extraneous!
        %plus_force_fib_i = hask.apSSA(@"+#", %force_fib_i)
        %fib_i_plus_fib_i_minus_one = hask.apSSA(%plus_force_fib_i, %force_fib_i_minus_one)
        hask.return(%fib_i_plus_fib_i_minus_one) }]
      [0 -> { ^entry(%default_random_name: !hask.untyped):
        hask.return(%i) }]
      [1 -> { ^entry(%default_random_name: !hask.untyped):
        hask.return(%i) }]
      hask.return(%retval)
    }
    hask.return(%lambda)
  }
  hask.dummy_finish
}
