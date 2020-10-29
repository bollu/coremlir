-- matmul that uses custom MLIR instructions to lower efficiently
-- import qualified Data.Vector.Unboxed as V

-- Fast
data Vector = Vector [Int] | VAdd Vector Vector | VMul Vector Vector | VSum Vector deriving(Show)
a, x, b :: Vector 

a = Vector [1, 2, 3, 4, 5, 6, 7, 8, 9]
x = Vector [3, 1, 4, 1, 5, 1, 6, 1, 7]
b = Vector [10, 20, 30, 40, 50, 60, 70]

outv = VAdd (VMul a x) b
outf = VSum outv

-- Slow
{-
-- a * x + b
a, x, b :: Vector Int
a = fromList [1, 2, 3, 4, 5, 6, 7, 8, 9]
x = fromList [3, 1, 4, 1, 5, 1, 6, 1, 7]
b = fromList [10, 20, 30, 40, 50, 60, 70]

outv = V.zipWith (+) (V.zipWith (*) a x) b
outf = V.foldl (+) 0 outv

main :: IO ()
main = print outv >> print outf
-}
