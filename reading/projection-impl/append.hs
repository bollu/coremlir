(append [xs zs]
  (case xs
    zs 
    y ys (CONS y (append ys zs))))

