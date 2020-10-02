
// addOption :: OptionInt -> OptionInt -> OptionInt
// addOption ox oy = 
//   case ox of
//     None -> None
//     Some x -> case oy of
//                 None -> None
//                 Some y -> Some (x + y)
//         
// 
// eval :: Expr -> OptionInt
// eval e = 
//   case e of
//     Add f1 f2 -> addOption (eval f1) (eval f2)
//     Val i -> Some i
//     Throw -> None
//
// ======================
// addWorker :: Int -> Int -> Int
// addWorker x y = x + y
//
// addOption :: OptionInt -> OptionInt -> OptionInt
// addOption ox oy =
//  case ox of
//    None -> None
//    Some x -> case oy of 
//                None -> None
//                Some y -> addWorker x y
//
//
// eval :: Expr -> OptionInt
// eval e = 
//   case e of
//     Add f1 f2 -> addOption (eval f1) (eval f2)
//     Val i -> Some i
//     Throw -> None
