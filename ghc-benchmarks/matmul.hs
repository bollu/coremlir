import qualified Data.Vector.Unboxed as V
import GHC.List as L


-- a * x + b
av, xv, bv :: V.Vector Int
av = V.replicate 10 1-- fromList [1, 2, 3, 4, 5, 6, 7, 8, 9]
xv = V.replicate 10 2
bv = V.replicate 10 3

outv :: V.Vector Int;
outv = V.zipWith (+) (V.zipWith (*) av xv) bv

main :: IO ()
mainv = print (V.sum outv)

al, xl, bl :: [Int]
al = L.replicate 10 1
xl = L.replicate 10 2
bl = L.replicate 10 2

outl :: [Int]
outl = zipWith (+) (zipWith (*) al xl) bl
mainl = print $ Prelude.sum (outl)

main = mainv >> mainl
