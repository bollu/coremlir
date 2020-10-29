import Data.Vector.Unboxed as V


-- a * x + b
a, x, b :: Vector Int
a = fromList [1, 2, 3, 4, 5, 6, 7, 8, 9]
x = fromList [3, 1, 4, 1, 5, 1, 6, 1, 7]
b = fromList [10, 20, 30, 40, 50, 60, 70]

outv = V.zipWith (+) (V.zipWith (*) a x) b
outf = V.sum 0 outv

main :: IO ()
main = print outv >> print outf
