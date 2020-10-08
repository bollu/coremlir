(before [xs] 
  (case xs
    NIL ;; NIL case
    y ys (if (= y 0) NIL (CONS y (before ys)))))

