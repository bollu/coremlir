{-# LANGUAGE CPP #-}
module Core2MLIR.ConvertToMLIR where
import Var (Var)
import qualified Var
import Id (isFCallId)
import Module (ModuleName, moduleNameFS, moduleName, pprModuleName)
import Unique (Unique, getUnique, unpkUnique)
import Name (getOccName, occNameFS, OccName, getName, nameModule_maybe)
import qualified BasicTypes as OccInfo (OccInfo(..), isStrongLoopBreaker)
import qualified CoreSyn
import CoreSyn (Expr(..), CoreExpr, Bind(..), CoreAlt, CoreBind, AltCon(..),)
import TyCoRep as Type (Type(..))
import Outputable (ppr, showSDoc, SDoc, vcat, hcat, text, hsep, nest, (<+>),
                   ($+$), hang, (<>), ($$), blankLine, lparen, rparen,
                   lbrack, rbrack, pprWithCommas, empty, comma)
import HscTypes (ModGuts(..))
import Module (ModuleName, moduleNameFS, moduleName)
import Control.Monad (ap, forM_)
import FastString
import Literal
import qualified Data.ByteString.Char8 as BS
 
-- import Text.PrettyPrint.ANSI.Leijen
-- https://hackage.haskell.org/package/ghc-8.10.1/docs/Outputable.html#v:SDoc
-- https://hackage.haskell.org/package/ghc-8.10.1/docs/src/Pretty.html#Doc



-- | name of the expression
type SSAName = SDoc


-- | monad instance
data Builder a = Builder { runBuilder_ :: Int -> (Int, a, SDoc) }

runBuilder :: Builder a -> (Int, a, SDoc)
runBuilder b = runBuilder_ b 0

instance Monad Builder where
    return a = Builder $ \i -> (i, a, empty)
    builda >>= a2buildb = Builder $ \i0 -> let (i1, a, doc1) = runBuilder_ builda i0
                                               (i2, b, doc2) = runBuilder_ (a2buildb a) i1 
                                           in (i2, b, doc1 $+$ doc2)

appendLine :: SDoc -> Builder ()
appendLine s = Builder $ \i -> (i, (), s)


instance Applicative Builder where pure = return; (<*>) = ap;
instance Functor Builder where fmap f mx = mx >>= (return . f)


(><) :: SDoc -> SDoc -> SDoc
(><) = (Outputable.<>)


braces_scoped :: SDoc -- ^ header
  -> SDoc -- ^ body 
  -> SDoc
braces_scoped header sdoc = (header <+> (text "{"))  $+$ (nest 2 sdoc) $+$ (text "}")

comment :: SDoc -> SDoc; comment s = hsep [text "//", s]
-- A 'FastString' is an array of bytes, hashed to support fast O(1)
-- comparison.  It is also associated with a character encoding, so that
-- Module.moduleName is a 'FastStrring'. How does one build a 'String'
-- from it?
cvtModuleToMLIR :: String -> ModGuts -> SDoc
cvtModuleToMLIR phase guts =
  let doc_name = pprModuleName $ Module.moduleName $ mg_module guts 
  in vcat [comment doc_name,
           comment (text phase),
             (braces_scoped (text "hask.module") $ 
                (vcat $ [cvtTopBind b | b <- mg_binds guts] ++ [text "hask.dummy_finish"]))]

recBindsScope :: SDoc
recBindsScope = text "hask.recursive_ref"


cvtBindRhs :: CoreExpr -> SDoc
cvtBindRhs rhs = 
  let (_, rhs_name, rhs_preamble) = runBuilder_ (flattenExpr rhs) 0
      body = rhs_preamble $+$ ((text "hask.return(") >< rhs_name >< (text ")"))  
  in body


cvtVar :: Var -> SDoc
cvtVar v = 
  let  varToUniqueName :: Var -> String
       -- varToUniqueName v = unpackFS $ occNameFS $ getOccName v
       varToUniqueName v = (escapeName  $ unpackFS $ occNameFS $ getOccName v) ++ "_" ++ (show $ getUnique v)

       -- | this is completely broken. 
       escapeName :: String -> String
       escapeName "-#" = "minus_hash"
       escapeName "+#" = "plus_hash"
       escapeName "()" = "unit_tuple"
       escapeName s = s -- error $ "unknown string (" ++ s ++ ")"
  in (text "%var__X_") >< (text $ varToUniqueName $ v) >< (text "_X_")


cvtTopBind :: CoreBind -> SDoc
cvtTopBind (NonRec b e) = 
    ((cvtVar b) <+> (text "=")) $$ 
    (nest 2 $ (text "hask.toplevel_binding") <+> (text "{") $$ (nest 2 $  (cvtBindRhs e)) $$ (text "}"))
cvtTopBind (Rec bs) = 
    braces_scoped (recBindsScope)
      (vcat $ [hsep [cvtVar b, text "=",
               braces_scoped (text "hask.toplevel_binding") (cvtBindRhs e)] | (b, e) <- bs])


parenthesize :: SDoc -> SDoc
parenthesize sdoc = lparen >< sdoc >< rparen


cvtLit :: Literal -> SDoc
cvtLit l =
    case l of
#if MIN_VERSION_ghc(8,8,0)
      Literal.LitChar x ->  ppr x
      Literal.LitString x -> ppr x
      Literal.LitNullAddr -> error $ "unknown: how to handle null addr cvtLit(Literal.LitNullAddr)?"
      Literal.LitFloat x -> ppr x
      Literal.LitDouble x -> ppr x
      Literal.LitLabel x _ _ -> ppr $ unpackFS  x
      Literal.LitRubbish ->  error $ "unknown: how to handle null addr cvtLit(Literal.LitRubbish)?"
#else
      Literal.MachChar x -> ppr x
      -- "UNHANDLED_MACH_STR" -- error $ "unknown: how to handle null addr cvtLit(Literal.MachStr)?" -- ppr x
      Literal.MachStr x -> text "hask.make_string(" ><  text "\"" >< text (BS.unpack x) >< text "\"" >< text ")"  
      Literal.MachNullAddr -> error $ "unknown: how to handle null addr cvtLit(Literal.LitNullAddr)?"
      Literal.MachFloat x -> error $ "unknown: how to handle null addr cvtLit(Literal.MachFloat)?"
      Literal.MachDouble x -> error $ "unknown: how to handle null addr cvtLit(Literal.MachDouble)?"
      Literal.MachLabel x _ _ -> ppr $ unpackFS  x
#endif
#if MIN_VERSION_ghc(8,6,0)
      Literal.LitNumber numty n _ ->
        case numty of
          Literal.LitNumInt -> text "hask.make_i32(" >< ppr n >< text ")"
          Literal.LitNumInt64 -> ppr n
          Literal.LitNumWord -> ppr n
          Literal.LitNumWord64 -> ppr n
          Literal.LitNumInteger -> ppr n
          Literal.LitNumNatural -> ppr n
#else
      Literal.MachInt x -> ppr x
      Literal.MachInt64 x -> ppr x
      Literal.MachWord x -> ppr x
      Literal.MachWord64 x -> ppr x
      Literal.LitInteger x _ -> ppr x
#endif

-- NOTE: I can't use this due to the stupid "add a suffix #" rule
-- cvtLit :: Literal -> SDoc; cvtLit l = pprLiteral id l 

-- | when converting an alt LHS, we want to print out the raw number, not
-- something like hask.make_i32(number).
-- * We want [0 -> ...], not [hask.make_i32(0) -> ]
-- * So we write another custom printer. Indeed, this is sick.
cvtAltLHSLit :: Literal -> SDoc
cvtAltLHSLit l =
    case l of
#if MIN_VERSION_ghc(8,8,0)
      Literal.LitChar x ->  ppr x
      Literal.LitString x -> ppr x
      Literal.LitNullAddr -> error $ "unknown: how to handle null addr cvtLit(Literal.LitNullAddr)?"
      Literal.LitFloat x -> ppr x
      Literal.LitDouble x -> ppr x
      Literal.LitLabel x _ _ -> ppr $ unpackFS  x
      Literal.LitRubbish ->  error $ "unknown: how to handle null addr cvtLit(Literal.LitRubbish)?"
#else
      Literal.MachChar x -> ppr x
      Literal.MachStr x -> text "UNHANDLED_MACH_STR" -- error $ "unknown: how to handle null addr cvtLit(Literal.MachStr)?" -- ppr x
      Literal.MachNullAddr -> error $ "unknown: how to handle null addr cvtLit(Literal.LitNullAddr)?"
      Literal.MachFloat x -> error $ "unknown: how to handle null addr cvtLit(Literal.MachFloat)?"
      Literal.MachDouble x -> error $ "unknown: how to handle null addr cvtLit(Literal.MachDouble)?"
      Literal.MachLabel x _ _ -> ppr $ unpackFS  x
#endif
#if MIN_VERSION_ghc(8,6,0)
      Literal.LitNumber numty n _ ->
        case numty of
          Literal.LitNumInt -> ppr n
          Literal.LitNumInt64 -> ppr n
          Literal.LitNumWord -> ppr n
          Literal.LitNumWord64 -> ppr n
          Literal.LitNumInteger -> ppr n
          Literal.LitNumNatural -> ppr n
#else
      Literal.MachInt x -> ppr x
      Literal.MachInt64 x -> ppr x
      Literal.MachWord x -> ppr x
      Literal.MachWord64 x -> ppr x
      Literal.LitInteger x _ -> ppr x
#endif
cvtAltLhs :: CoreSyn.AltCon -> SDoc
cvtAltLhs (DataAlt altcon) = text "DATACONSTRUCTOR" -- Ast.AltDataCon $ occNameToText $ getOccName altcon
cvtAltLhs (LitAlt l)       = cvtAltLHSLit l
cvtAltLhs DEFAULT          = text "\"default\""


arrow :: SDoc; arrow = text "->"

cvtAltRHS :: Var -> [Var] -> CoreExpr -> Builder ()
cvtAltRHS wild bnds rhs = Builder $ \i0 -> 
 -- | HACK: we need to start from something other than 100...
 let (i1, name_rhs, preamble_rhs) = runBuilder_ (flattenExpr rhs) i0
     params = pprWithCommas (\v -> cvtVar v ><  text ": none") (wild:bnds)
     inner = (text "^entry(" >< params >< text "):") $$ (nest 2 $ preamble_rhs $$ ((text "hask.return(") >< name_rhs >< (text ")")))
 in (i1, (), (text "{") $$ (nest 2 inner) $$ (text "}"))

cvtAlt :: Var -> CoreAlt -> Builder ()
cvtAlt wild (lhs, bs, e) = Builder $ \i0 -> 
                    let (i1, (), rhs) = runBuilder_ (cvtAltRHS wild bs e)i0
                    in (i1, (), (lbrack >< cvtAltLhs lhs <+> arrow) $$ (nest 2 $ rhs  >< rbrack))



flattenExpr :: CoreExpr -> Builder SSAName
flattenExpr expr =
  case expr of
    Var x -> return (cvtVar x)--return ((text"%") >< ppr x)
    Lam param body -> 
     Builder $ \i0 ->
      let (i1, name_body, preamble_body) = runBuilder_ (flattenExpr body) i0
          name_lambda = text ("%lambda_" ++ show i1)
          fulldoc =  (name_lambda) <+> (text "=") $$
                         (nest 2 $ ((text "hask.lambdaSSA(") >< ((text "%") >< (ppr param)) >< (text ")") <+> (text "{")) $$ 
                            (nest 2 (preamble_body $+$ ((text "hask.return(") >< name_body >< (text ")")))) $$
                            text "}")
          in (i1+1, name_lambda, fulldoc)
    Case scrutinee wild _ as -> Builder $ \i0 -> 
                                  let (i1, name_scrutinee, preamble_scrutinee) = runBuilder_ (flattenExpr scrutinee) i0
                                      name_case = text ("%case_" ++ show i1) 
                                      (i2, _, alts_doc) = runBuilder_ (forM_ as (cvtAlt wild)) i1
                                      fulldoc = preamble_scrutinee $+$ 
                                              hang ((name_case <+>  (text "=") $+$ (nest 2 $ (text "hask.caseSSA") <+> name_scrutinee)))
                                                    2
                                                    alts_doc
                                  in (i2+1, name_case, fulldoc)

    App f x -> Builder $ \i0 ->
                let (i1, name_f, preamble_f) = runBuilder_ (flattenExpr f) i0
                    (i2, name_x, preamble_x) = runBuilder_ (flattenExpr x) i1
                    name_app = text ("%app_" ++ show i2)
                    fulldoc = preamble_f $+$ preamble_x $+$ (name_app <+> (text " = ") <+> (text "hask.apSSA(") >< name_f >< comma <+> name_x >< (text ")"))
                in (i2+1, name_app, fulldoc)
    Lit l -> Builder $ \i0 ->
               let  name_lit = text $ "%lit_" ++ show i0
                    fulldoc =  name_lit <+> (text " = ") <+>  cvtLit l
               in (i0+1, name_lit, fulldoc)
          -- return (text ("LITERAL"))
    Type t -> Builder $ \i0 ->
              let type_lit = text $ "%type_" ++ show i0 -- text $ "hask.make_string(\"TYPEINFO_ERASED\")" 
                  fulldoc = type_lit <+> (text " = ") <+> (text "hask.make_string(\"TYPEINFO_ERASED\")")
              in (i0+1, type_lit, fulldoc)
              -- return $ text  "hask.make_string(\"TYPEINFO_ERASED\")" -- (text ("TYPE"))
    Tick _ e -> return (text ("TICK"))
    Cast _ e -> return (text ("CAST"))

    _ -> Builder $ \i0 -> let name_unimpl = text ("%unimpl_" ++ show i0)
                              fulldoc = name_unimpl <+> (text " = ") <+> (text "hask.make_i32(42)") 
                        in (i0+1, name_unimpl, fulldoc)

-- instantiates an expression, giving it a name and an SDoc that needs to be pasted above it.
-- TODO: we need a monad here to allow us to build an AST while returning a variable name.

-- cvtExpr :: CoreExpr -> SDoc
-- cvtExpr expr =
--   case expr of
--     Var x -> text "%" >< ppr x
--     Lam x e -> braces_scoped (text "hask.lambda" <+> (parenthesize (cvtVar x))) (cvtExpr e)
--     Case e wild _ as -> text "hask.caseSSA" <+> (cvtExpr e)  $+$ (nest 2 $ vcat [cvtAlt wild a | a <-as ])
--     _ -> text  "hask.dummy_finish"

  -- case expr of
  --   Var x
  --       -- foreign calls are local but have no binding site.
  --       -- TODO: use hasNoBinding here.
  --     | isFCallId x   -> EVarGlobal ForeignCall
  --     | Just m <- nameModule_maybe $ getName x
  --                     -> EVarGlobal $ ExternalName (cvtModuleName $ Module.moduleName m)
  --                                                  (occNameToText $ getOccName x)
  --                                                  (cvtUnique $ getUnique x)
  --     | otherwise     -> EVar (cvtVar x)
  --   Lit l             -> ELit (cvtLit l)
  --   App x y           -> EApp (cvtExpr x) (cvtExpr y)
  --   Lam x e
  --     | Var.isTyVar x -> ETyLam (cvtVar x) (cvtExpr e)
  --     | otherwise     -> ELam (cvtBinder x) (cvtExpr e)
  --   Let (NonRec b e) body -> ELet [(cvtBinder b, cvtExpr e)] (cvtExpr body)
  --   Let (Rec bs) body -> ELet (map (bimap cvtBinder cvtExpr) bs) (cvtExpr body)
  --   Case e x _ as     -> ECase (cvtExpr e) (cvtBinder x) (map cvtAlt as)
  --   Cast x _          -> cvtExpr x
  --   Tick _ e          -> cvtExpr e
  --   Type t            -> EType $ cvtType t
  --   Coercion _        -> ECoercion


{-
import Data.Bifunctor
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE

import Literal (Literal(..))
#if MIN_VERSION_ghc(8,6,0)
import qualified Literal
#endif
import Var (Var)
import qualified Var
import Id (isFCallId)
import Module (ModuleName, moduleNameFS, moduleName)
import Unique (Unique, getUnique, unpkUnique)
import Name (getOccName, occNameFS, OccName, getName, nameModule_maybe)
import qualified IdInfo
import qualified BasicTypes as OccInfo (OccInfo(..), isStrongLoopBreaker)
#if MIN_VERSION_ghc(8,0,0)
import qualified CoreStats
#else
import qualified CoreUtils as CoreStats
#endif
import qualified CoreSyn
import CoreSyn (Expr(..), CoreExpr, Bind(..), CoreAlt, CoreBind, AltCon(..))
import HscTypes (ModGuts(..))
import FastString (FastString)
import qualified FastString
#if MIN_VERSION_ghc(8,2,0)
import TyCoRep as Type (Type(..))
#elif MIN_VERSION_ghc(8,0,0)
import TyCoRep as Type (Type(..), TyBinder(..))
#else
import TypeRep as Type (Type(..))
#endif
#if !(MIN_VERSION_ghc(8,2,0))
import Type (splitFunTy_maybe)
#endif
import TyCon (TyCon, tyConUnique)

import Outputable (ppr, showSDoc, SDoc)
import DynFlags (unsafeGlobalDynFlags)

import Core2MLIR.Ast as Ast

cvtSDoc :: SDoc -> T.Text
cvtSDoc = T.pack . showSDoc unsafeGlobalDynFlags

fastStringToText :: FastString -> T.Text
fastStringToText = TE.decodeUtf8
#if MIN_VERSION_ghc(8,10,0)
  . FastString.bytesFS
#else
  . FastString.fastStringToByteString
#endif

occNameToText :: OccName -> T.Text
occNameToText = fastStringToText . occNameFS

cvtUnique :: Unique.Unique -> Ast.Unique
cvtUnique u =
    let (a,b) = unpkUnique u
    in Ast.Unique a b

cvtVar :: Var -> BinderId
cvtVar = BinderId . cvtUnique . Var.varUnique

cvtBinder :: Var -> SBinder
cvtBinder v
  | Var.isId v =
    SBndr $ Binder { binderName   = occNameToText $ getOccName v
                   , binderId     = cvtVar v
                   , binderIdInfo = cvtIdInfo $ Var.idInfo v
                   , binderIdDetails = cvtIdDetails $ Var.idDetails v
                   , binderType   = cvtType $ Var.varType v
                   }
  | otherwise =
    SBndr $ TyBinder { binderName   = occNameToText $ getOccName v
                     , binderId     = cvtVar v
                     , binderKind   = cvtType $ Var.varType v
                     }

cvtIdInfo :: IdInfo.IdInfo -> Ast.IdInfo SBinder BinderId
cvtIdInfo i =
    IdInfo { idiArity         = IdInfo.arityInfo i
           , idiIsOneShot     = IdInfo.oneShotInfo i == IdInfo.OneShotLam
           , idiUnfolding     = cvtUnfolding $ IdInfo.unfoldingInfo i
           , idiInlinePragma  = cvtSDoc $ ppr $ IdInfo.inlinePragInfo i
           , idiOccInfo       = case IdInfo.occInfo i of
#if MIN_VERSION_ghc(8,2,0)
                                  OccInfo.ManyOccs{} -> OccManyOccs
#else
                                  OccInfo.NoOccInfo  -> OccManyOccs
#endif
                                  OccInfo.IAmDead    -> OccDead
                                  OccInfo.OneOcc{}   -> OccOneOcc
                                  oi@OccInfo.IAmALoopBreaker{} -> OccLoopBreaker (OccInfo.isStrongLoopBreaker oi)
           , idiStrictnessSig = cvtSDoc $ ppr $ IdInfo.strictnessInfo i
           , idiDemandSig     = cvtSDoc $ ppr $ IdInfo.demandInfo i
           , idiCallArity     = IdInfo.callArityInfo i
           }

cvtUnfolding :: CoreSyn.Unfolding -> Ast.Unfolding SBinder BinderId
cvtUnfolding CoreSyn.NoUnfolding = Ast.NoUnfolding
#if MIN_VERSION_ghc(8,2,0)
cvtUnfolding CoreSyn.BootUnfolding = Ast.BootUnfolding
#endif
cvtUnfolding (CoreSyn.OtherCon cons) = Ast.OtherCon (map cvtAltCon cons)
cvtUnfolding (CoreSyn.DFunUnfolding{}) = Ast.DFunUnfolding
cvtUnfolding u@(CoreSyn.CoreUnfolding{}) =
    Ast.CoreUnfolding { unfTemplate   = cvtExpr $ CoreSyn.uf_tmpl u
                      , unfIsValue    = CoreSyn.uf_is_value u
                      , unfIsConLike  = CoreSyn.uf_is_conlike u
                      , unfIsWorkFree = CoreSyn.uf_is_work_free u
                      , unfGuidance   = cvtSDoc $ ppr $ CoreSyn.uf_guidance u
                      }

cvtIdDetails :: IdInfo.IdDetails -> Ast.IdDetails
cvtIdDetails d =
    case d of
      IdInfo.VanillaId -> Ast.VanillaId
      IdInfo.RecSelId{} -> Ast.RecSelId
      IdInfo.DataConWorkId{} -> Ast.DataConWorkId
      IdInfo.DataConWrapId{} -> Ast.DataConWrapId
      IdInfo.ClassOpId{} -> Ast.ClassOpId
      IdInfo.PrimOpId{} -> Ast.PrimOpId
      IdInfo.FCallId{} -> error "This shouldn't happen"
      IdInfo.TickBoxOpId{} -> Ast.TickBoxOpId
      IdInfo.DFunId{} -> Ast.DFunId
#if MIN_VERSION_ghc(8,0,0)
      IdInfo.CoVarId{} -> Ast.CoVarId
#endif
#if MIN_VERSION_ghc(8,2,0)
      IdInfo.JoinId n -> Ast.JoinId n
#endif

cvtCoreStats :: CoreStats.CoreStats -> Ast.CoreStats
cvtCoreStats stats =
    Ast.CoreStats
      { csTerms     = CoreStats.cs_tm stats
      , csTypes     = CoreStats.cs_ty stats
      , csCoercions = CoreStats.cs_co stats
#if MIN_VERSION_ghc(8,2,0)
      , csValBinds  = CoreStats.cs_vb stats
      , csJoinBinds = CoreStats.cs_jb stats
#else
      , csValBinds  = 0
      , csJoinBinds = 0
#endif
      }

exprStats :: CoreExpr -> CoreStats.CoreStats
#if MIN_VERSION_ghc(8,0,0)
exprStats = CoreStats.exprStats
#else
-- exprStats wasn't exported in 7.10
exprStats _ = CoreStats.CS 0 0 0
#endif

cvtTopBind :: CoreBind -> STopBinding
cvtTopBind (NonRec b e) =
    NonRecTopBinding (cvtBinder b) (cvtCoreStats $ exprStats e) (cvtExpr e)
cvtTopBind (Rec bs) =
    RecTopBinding $ map to bs
  where to (b, e) = (cvtBinder b, cvtCoreStats $ exprStats e, cvtExpr e)

cvtExpr :: CoreExpr -> Ast.SExpr
cvtExpr expr =
  case expr of
    Var x
        -- foreign calls are local but have no binding site.
        -- TODO: use hasNoBinding here.
      | isFCallId x   -> EVarGlobal ForeignCall
      | Just m <- nameModule_maybe $ getName x
                      -> EVarGlobal $ ExternalName (cvtModuleName $ Module.moduleName m)
                                                   (occNameToText $ getOccName x)
                                                   (cvtUnique $ getUnique x)
      | otherwise     -> EVar (cvtVar x)
    Lit l             -> ELit (cvtLit l)
    App x y           -> EApp (cvtExpr x) (cvtExpr y)
    Lam x e
      | Var.isTyVar x -> ETyLam (cvtBinder x) (cvtExpr e)
      | otherwise     -> ELam (cvtBinder x) (cvtExpr e)
    Let (NonRec b e) body -> ELet [(cvtBinder b, cvtExpr e)] (cvtExpr body)
    Let (Rec bs) body -> ELet (map (bimap cvtBinder cvtExpr) bs) (cvtExpr body)
    Case e x _ as     -> ECase (cvtExpr e) (cvtBinder x) (map cvtAlt as)
    Cast x _          -> cvtExpr x
    Tick _ e          -> cvtExpr e
    Type t            -> EType $ cvtType t
    Coercion _        -> ECoercion

cvtAlt :: CoreAlt -> Ast.SAlt
cvtAlt (con, bs, e) = Alt (cvtAltCon con) (map cvtBinder bs) (cvtExpr e)

cvtAltCon :: CoreSyn.AltCon -> Ast.AltCon
cvtAltCon (DataAlt altcon) = Ast.AltDataCon $ occNameToText $ getOccName altcon
cvtAltCon (LitAlt l)       = Ast.AltLit $ cvtLit l
cvtAltCon DEFAULT          = Ast.AltDefault

cvtLit :: Literal -> Ast.Lit
cvtLit l =
    case l of
#if MIN_VERSION_ghc(8,8,0)
      Literal.LitChar x -> Ast.MachChar x
      Literal.LitString x -> Ast.MachStr x
      Literal.LitNullAddr -> Ast.MachNullAddr
      Literal.LitFloat x -> Ast.MachFloat x
      Literal.LitDouble x -> Ast.MachDouble x
      Literal.LitLabel x _ _ -> Ast.MachLabel $ fastStringToText  x
      Literal.LitRubbish -> Ast.LitRubbish
#else
      Literal.MachChar x -> Ast.MachChar x
      Literal.MachStr x -> Ast.MachStr x
      Literal.MachNullAddr -> Ast.MachNullAddr
      Literal.MachFloat x -> Ast.MachFloat x
      Literal.MachDouble x -> Ast.MachDouble x
      Literal.MachLabel x _ _ -> Ast.MachLabel $ fastStringToText  x
#endif
#if MIN_VERSION_ghc(8,6,0)
      Literal.LitNumber numty n _ ->
        case numty of
          Literal.LitNumInt -> Ast.MachInt n
          Literal.LitNumInt64 -> Ast.MachInt64 n
          Literal.LitNumWord -> Ast.MachWord n
          Literal.LitNumWord64 -> Ast.MachWord64 n
          Literal.LitNumInteger -> Ast.LitInteger n
          Literal.LitNumNatural -> Ast.LitNatural n
#else
      Literal.MachInt x -> Ast.MachInt x
      Literal.MachInt64 x -> Ast.MachInt64 x
      Literal.MachWord x -> Ast.MachWord x
      Literal.MachWord64 x -> Ast.MachWord64 x
      Literal.LitInteger x _ -> Ast.LitInteger x
#endif

cvtModule :: String -> ModGuts -> Ast.SModule
cvtModule phase guts =
    Ast.Module name (T.pack phase) (map cvtTopBind $ mg_binds guts)
  where name = cvtModuleName $ Module.moduleName $ mg_module guts

cvtModuleName :: Module.ModuleName -> Ast.ModuleName
cvtModuleName = Ast.ModuleName . fastStringToText . moduleNameFS

cvtType :: Type.Type -> Ast.SType
#if MIN_VERSION_ghc(8,10,0)
cvtType (Type.FunTy _flag a b) = Ast.FunTy (cvtType a) (cvtType b)
#elif MIN_VERSION_ghc(8,2,0)
cvtType (Type.FunTy a b) = Ast.FunTy (cvtType a) (cvtType b)
#else
cvtType t
  | Just (a,b) <- splitFunTy_maybe t = Ast.FunTy (cvtType a) (cvtType b)
#endif
cvtType (Type.TyVarTy v)       = Ast.VarTy (cvtVar v)
cvtType (Type.AppTy a b)       = Ast.AppTy (cvtType a) (cvtType b)
cvtType (Type.TyConApp tc tys) = Ast.TyConApp (cvtTyCon tc) (map cvtType tys)
#if MIN_VERSION_ghc(8,8,0)
cvtType (Type.ForAllTy (Var.Bndr b _) t) = Ast.ForAllTy (cvtBinder b) (cvtType t)
#elif MIN_VERSION_ghc(8,2,0)
cvtType (Type.ForAllTy (Var.TvBndr b _) t) = Ast.ForAllTy (cvtBinder b) (cvtType t)
#elif MIN_VERSION_ghc(8,0,0)
cvtType (Type.ForAllTy (Named b _) t) = Ast.ForAllTy (cvtBinder b) (cvtType t)
cvtType (Type.ForAllTy (Anon _) t)    = cvtType t
#else
cvtType (Type.ForAllTy b t)    = Ast.ForAllTy (cvtBinder b) (cvtType t)
#endif
cvtType (Type.LitTy _)         = Ast.LitTy
#if MIN_VERSION_ghc(8,0,0)
cvtType (Type.CastTy t _)      = cvtType t
cvtType (Type.CoercionTy _)    = Ast.CoercionTy
#endif

cvtTyCon :: TyCon.TyCon -> Ast.TyCon
cvtTyCon tc = TyCon (occNameToText $ getOccName tc) (cvtUnique $ tyConUnique tc)
-}  
