// RUN: ../build/bin/hask-opt %s  -interpret | FileCheck %s
// RUN: ../build/bin/hask-opt %s  -interpret -worker-wrapper | FileCheck %s -check-prefix='CHECK-WW'
// RUN: ../build/bin/hask-opt %s -lower-std -lower-llvm | FileCheck %s || true
// RUN: ../build/bin/hask-opt %s  | ../build/bin/hask-opt -lower-std -lower-llvm |  FileCheck %s || true
// Check that @plus works with Maybe works.
// CHECK: constructor(Just 42)
// CHECK: num_thunkify_calls(38)
// CHECK: num_force_calls(76)
// CHECK: num_construct_calls(76)

// CHECK-WW: constructor(Just 42)
// CHECK-WW: num_thunkify_calls(0)
// CHECK-WW: num_force_calls(0)
// CHECK-WW: num_construct_calls(40)

module {
  // should it be Attr Attr, with the "list" embedded as an attribute,
  // or should it be Attr [Attr]? Who really knows :(
  // define the algebraic data type
  // TODO: setup constructors properly.
  hask.adt @Maybe [#hask.data_constructor<@Just [@"Int#"]>, #hask.data_constructor<@Nothing []>]

  // f :: Maybe -> Maybe
  // f i = case i of Maybe i# ->
  //        case i# of 0 -> Maybe 5;
  //        _ -> case f ( Maybe(i# -# 1#)) of
  //              Nothing -> Nothing
  //              Just j# -> Just (j# +# 1#)
  hask.func @f (%i : !hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe> {
      %icons = hask.force(%i): !hask.adt<@Maybe>
      %reti = hask.case @Maybe %icons 
           [@Nothing -> {
              %nothing = hask.construct(@Nothing): !hask.adt<@Maybe>
              hask.return (%nothing):!hask.adt<@Maybe>
           }
           [@Just -> { ^entry(%ihash: !hask.value):
              %retj = hask.caseint %ihash
                  [0 -> {
                        %five = hask.make_i64(5)
                        %boxed = hask.construct(@Just, %five:!hask.value): !hask.adt<@Maybe>
                        hask.return(%boxed) : !hask.adt<@Maybe>
                  }]
                  [@default ->  {
                        %one = hask.make_i64(1)
                        %isub = hask.primop_sub(%ihash, %one)
                        %boxed_isub = hask.construct(@Just, %isub: !hask.value): !hask.adt<@Maybe>
                        %boxed_isub_t = hask.thunkify(%boxed_isub : !hask.adt<@Maybe>) : !hask.thunk<!hask.adt<@Maybe>>
                        %f = hask.ref(@f): !hask.fn<(!hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe>>
                        %rec_t = hask.ap(%f : !hask.fn<(!hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe>> , %boxed_isub_t)
                        %rec_v = hask.force(%rec_t): !hask.adt<@Maybe>
			%out_v = hask.case @Maybe %rec_v 
			       [@Nothing -> { ^entry:
				  %nothing = hask.construct(@Nothing): !hask.adt<@Maybe>
				  hask.return (%nothing):!hask.adt<@Maybe>
			       }]

			       [@Just -> { ^entry(%jhash: !hask.value):
			       	  // TODO: worried about capture semantics!
				  %one_inner = hask.make_i64(1)
			          %jhash_incr =  hask.primop_add(%jhash, %one_inner)
				  %boxed_jash_incr = hask.construct(@Just, %jhash_incr: !hask.value): !hask.adt<@Maybe>
				  hask.return(%boxed_jash_incr):!hask.adt<@Maybe>
			       }]
                        hask.return(%out_v): !hask.adt<@Maybe>
                  }]
              hask.return(%retj):!hask.adt<@Maybe>
           }]
      hask.return(%reti): !hask.adt<@Maybe>
    }

  // 37 + 5 = 42
  hask.func@main () -> !hask.adt<@Maybe> {
      %v = hask.make_i64(37)
      %v_box = hask.construct(@Just, %v:!hask.value): !hask.adt<@Maybe>
      %v_thunk = hask.thunkify(%v_box: !hask.adt<@Maybe>): !hask.thunk<!hask.adt<@Maybe>>
      %f = hask.ref(@f): !hask.fn<(!hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe>>
      %out_t = hask.ap(%f : !hask.fn<(!hask.thunk<!hask.adt<@Maybe>>) -> !hask.adt<@Maybe>>, %v_thunk)
      %out_v = hask.force(%out_t): !hask.adt<@Maybe>
      hask.return(%out_v) : !hask.adt<@Maybe>
    }
}

