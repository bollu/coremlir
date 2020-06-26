// The syntax that I encoded into the parser
// %fib :: Int -> Int
func @fib () {
    %constructor_ihash  = standalone.make_data_constructor<"I#"> 

  standalone.lambda [%i] {
    standalone.case  {standalone.return(%i)} { alt0 = "default", alt1=0, alt2=1 }
    { //default
    }
    { // 0 -> 
        standalone.ap(
            { standalone.return (%constructor_ihash) },
            { %c0 = constant 0 : i32 standalone.make_i32 (%c0) }) 
    }
    { // 1 -> 
        standalone.ap(
            { standalone.return (%constructor_ihash) },
            { %c1 = constant 0 : i32 standalone.make_i32 (%c1) }) 

    }

  }
}