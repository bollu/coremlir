// Show that having the function body be a recursive-def does not
// allow us to write recursive calls


                      
hask.module { 
    %core_one =  hask.make_i32(1)

    // passes
    %flat = hask.recursive_ref  {  
        %fib_call = hask.apSSA(%flat, %core_one)
        hask.return(%fib_call)
    }

    // fails!
    %nested = hask.recursive_ref  {  
        %case = hask.caseSSA %core_one 
                ["default" -> { //default
                    // fails because the use is nested inside a region.
                    %fib_call = hask.apSSA(%nested, %core_one)
                    hask.return(%fib_call)
                }]
        hask.return(%case)
    }
    hask.dummy_finish
}

