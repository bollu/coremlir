(doubles xs
  (case xs
    NIL 
    y ys (CONS (* 2 y) (doubles ys))))
