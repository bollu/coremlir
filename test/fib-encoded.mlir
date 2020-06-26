// The syntax that I encoded into the parser
// %fib :: Int -> Int
func @fib () {
  standalone.lambda [%i] {
    standalone.case  {standalone.return(%i)} { alt0 = "default", alt1=0, alt2=1 }
    {
    }
    { // 0 -> 

    }
    { // 1 -> 

    }

  }
}