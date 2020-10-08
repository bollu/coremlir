(doubles [xs] 
  (case xs
    NIL ;; NIL case
    y ys (CONS (* 2 y) (doubles ys))))
