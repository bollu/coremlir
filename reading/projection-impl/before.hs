(before xs
  (case xs
    NIL 
    y ys (if (= y 0) NIL (CONS y (before ys)))))

