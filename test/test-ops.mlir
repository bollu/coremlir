// playground file to test op syntax
hask.module { 
// hask.dominance_free_scope {

%constructor_ihash  = hask.make_data_constructor<"I#"> 
// This is kind of a lie, we should call it as inbuilt fn or whatever.
%constructor_plus = hask.make_data_constructor<"GHC.Num.+"> 
%constructor_minus = hask.make_data_constructor<"GHC.Num.-"> 
%value_dict_num_int = hask.make_data_constructor<"GHC.Num.$fNumInt">

// The syntax that I encoded into the parser
// %fib :: Int -> Int
hask.func @fib {  
  hask.lambda (%i) {
        %f = hask.apSSA(@fib, %i)
    	%x = hask.caseSSA  %i
    		[ "default" -> { ^entry(%wild: none): hask.return (%i) }]
    	hask.return(%x)
    }	
}

// This is hopeless, we can have SSA values and symbol table entries with
// the same name.
hask.func @function {
    %function = hask.make_i32(1)
    hask.return (%function)
}

hask.dummy_finish
}  
