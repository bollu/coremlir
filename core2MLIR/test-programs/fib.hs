fib :: Int -> Int
fib x = case x of 0 -> 0; 1 -> 1; n -> fib (n-1) + fib (n-2)

main :: IO (); main = putStrLn (show (fib 10))
        
