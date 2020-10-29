{-# LANGUAGE CPP #-}
{-# LANGUAGE ViewPatterns #-}
module Core2MLIR.ConvertToMLIR(cvtModuleToMLIR) where
import Var (Var, varName)
import qualified Var
import Id (isFCallId, isGlobalId, isExportedId)
import Module (ModuleName, moduleNameFS, moduleName, pprModuleName)
import Unique (Unique, getUnique, unpkUnique)
import Name (getOccName, occNameFS, OccName, getName, nameModule_maybe, Name, nameStableString)
import qualified BasicTypes as OccInfo (OccInfo(..), isStrongLoopBreaker)
import qualified CoreSyn
import CoreSyn (Expr(..), CoreExpr, Bind(..), CoreAlt, CoreBind, AltCon(..),)
import TyCoRep as Type (Type(..))
import Outputable
-- import Outputable (ppr, showSDoc, SDoc, vcat, hcat, text, hsep, nest, (<+>),
--                    ($+$), hang, (<>), ($$), blankLine, lparen, rparen,
--                    lbrack, rbrack, pprWithCommas, empty, comma, renderWithStyle,
--                    defaultDumpStyle, punctuate, hsep, reallyAlwaysQualify,
--                    showSDocDump, showSDocDebug, initSDocContext, mkDumpStyle)
import PprCore (pprCoreBindingsWithSize)
import HscTypes (ModGuts(..))
import Module (ModuleName, moduleNameFS, moduleName)
import DataCon
import Control.Monad (ap, forM_)
import TyCon
import FastString
import Literal
import Control.Monad.State
import qualified Data.Set as S
import qualified Data.ByteString.Char8 as BS
import qualified Core2MLIR.MLIR as MLIR
import GHC(DynFlags, TyCon)

-- import Text.PrettyPrint.ANSI.Leijen
-- https://hackage.haskell.org/package/ghc-8.10.1/docs/Outputable.html#v:SDoc
-- https://hackage.haskell.org/package/ghc-8.10.1/docs/src/Pretty.html#Doc

docDoubleQuote :: SDoc
docDoubleQuote = text "\""

-- | name of the expression
type SSAName = SDoc

-- | keep around stuff that can be recursive, so we can emit them 
-- differently.
type PossibleRecursiveVar = Var

-- | monad instance
data Builder a = Builder { runBuilder_ :: (Int, S.Set PossibleRecursiveVar) -> ((Int, S.Set PossibleRecursiveVar), a, SDoc) } -- , recnames :: S.Set String }

runBuilder :: Builder () -> SDoc
runBuilder b = let (_, _, doc) = runBuilder_ b (0, S.empty) in doc

instance Monad Builder where
    return a = Builder $ \state -> (state, a, empty)
    builda >>= a2buildb = 
      Builder $ \state0 ->
        let (state1, a, doc1) = runBuilder_ builda state0
            (state2, b, doc2) = runBuilder_ (a2buildb a) state1
        in (state2, b, doc1 $+$ doc2)

-- | FFS. Some days I hate what I do. If I build stuff like this, then my 
-- line concatenation algo is wrong. *sigh*
builderAppend :: SDoc -> Builder ()
builderAppend s = Builder $ \state -> (state, (), s)

builderMakeUnique :: Builder Int
builderMakeUnique = Builder $ \(i, vars) -> ((i+1, vars), i, empty)

builderNest :: Int -> Builder a -> Builder a
builderNest depth b = Builder $ \state0 -> 
  let (state1, a, doc1) = runBuilder_ b state0
  in (state1, a, nest depth doc1)

-- | bracket the use of a recursive var, adding the variable to names.
builderBracketRecursiveVar :: PossibleRecursiveVar -> Builder a -> Builder a
builderBracketRecursiveVar newv builda = 
    Builder $ \(i0, vars0) -> 
        let ((i1, vars1), a, doc1) = runBuilder_ builda (i0, S.insert newv vars0)
        in ((i1, vars0), a, doc1)

builderBracketRecursiveVars :: [PossibleRecursiveVar] -> Builder a -> Builder a
builderBracketRecursiveVars newvars builda = 
    Builder $ \(i0, vars0) -> 
        let ((i1, vars1), a, doc1) = runBuilder_ builda (i0, (S.fromList newvars) `S.union` vars0)
        in ((i1, vars0), a, doc1)

-- | Check if the given variable is recursive
builderIsVarRecursive :: PossibleRecursiveVar -> Builder Bool
builderIsVarRecursive v = Builder $ \(i0, vars0) -> ((i0, vars0), S.member v vars0, empty)


instance Applicative Builder where pure = return; (<*>) = ap;
instance Functor Builder where fmap f mx = mx >>= (return . f)


(><) :: SDoc -> SDoc -> SDoc
(><) = (Outputable.<>)


braces_scoped :: SDoc -- ^ header
  -> SDoc -- ^ body 
  -> SDoc
braces_scoped header sdoc = (header <+> (text "{"))  $+$ (nest 2 sdoc) $+$ (text "}")

comment :: SDoc -> SDoc; comment s = hsep [text "//", s]


intercalateCommentsInString :: String -> String
intercalateCommentsInString [] = []
intercalateCommentsInString ('\n':xs) = "\n//" ++  intercalateCommentsInString xs
intercalateCommentsInString (x:xs) = x:intercalateCommentsInString xs

dumpProgramAsCore :: DynFlags -> ModGuts -> String
dumpProgramAsCore dflags guts = 
  let sdoc = pprCoreBindingsWithSize $ mg_binds guts
      string = renderWithStyle dflags sdoc ((mkDumpStyle dflags reallyAlwaysQualify)) 
  in "//" ++ intercalateCommentsInString string



-- | Returns true if the core bind is defining 
-- -- RHS size: {terms: 2, types: 1, coercions: 0, joins: 0/0}
-- :Main.main :: IO ()
-- [LclIdX]
-- :Main.main = GHC.TopHandler.runMainIO @ () main

-- Fuck this, I have no idea how to do this right.
-- I'm just going to
shouldKeepBind :: CoreBind -> Bool
shouldKeepBind _ = True
-- shouldKeepBind (NonRec var e) = 
--     let binderName =  unpackFS . occNameFS $ getOccName $ var
--     in False -- (binderName == "fib")
-- shouldKeepBind (Rec _) = True

mlirPrelude :: SDoc
mlirPrelude = 
  (text "hask.make_data_constructor @\"+#\"") $+$
  (text "hask.make_data_constructor @\"-#\"") $+$
  (text "hask.make_data_constructor @\"()\"")

-- A 'FastString' is an array of bytes, hashed to support fast O(1)
-- comparison.  It is also associated with a character encoding, so that
-- Module.moduleName is a 'FastStrring'. How does one build a 'String'
-- from it?
-- cvtModuleToMLIR :: DynFlags -> String -> ModGuts -> SDoc
-- cvtModuleToMLIR dfags phase guts =
--   let doc_name = pprModuleName $ Module.moduleName $ mg_module guts 
--   in vcat [comment doc_name,
--            comment (text phase),
--            (braces_scoped (text "hask.module") $ (mlirPrelude $+$ (vcat  $ [cvtTopBind b | b <- mg_binds guts, shouldKeepBind b] ++ [text "hask.dummy_finish"]))),
--            text $ "// ============ Haskell Core ========================",
--            text $ dumpProgramAsCore dfags guts]

                                                                
getBindLHSs :: CoreBind -> [Var]
getBindLHSs (NonRec b e) = [b]
getBindLHSs (Rec bs) = [ b | (b, _) <- bs]                                


cvtDataConToMLIR :: DataCon -> Builder ()
cvtDataConToMLIR dc@(isVanillaDataCon -> True) = do
  let (univex, theta, constructorTys, origResultTy) =   (dataConSig dc) 
  builderAppend $ text "  ==DATACON: " ><  ppr (dataConName dc) >< text "=="
  builderAppend $ text "  dcOrigTyCon: " ><  ppr (dataConOrigTyCon dc)
  builderAppend $ text "  dcFieldLabels: " ><  ppr (dataConFieldLabels dc)
  builderAppend $ text "  dcRepType: " ><  ppr (dataConRepType dc)
  builderAppend $ text "  constructor types: " ><  ppr  (constructorTys)
  builderAppend $ text "  result type: " ><  ppr  (origResultTy)
  builderAppend $ text "  ---"
  builderAppend $ text "  dcSig: " ><  ppr  (dataConSig dc)
  builderAppend $ text "  dcFullSig: " ><  ppr (dataConFullSig dc)
  builderAppend $ text "  dcUniverseTyVars: " >< ppr (dataConUnivTyVars dc)
  builderAppend $ text "  dcArgs: " >< ppr (dataConOrigArgTys dc)
  builderAppend $ text "  dcOrigArgTys: " ><  ppr (dataConOrigArgTys dc)
  builderAppend $ text "  dcOrigResTy: " ><  ppr (dataConOrigResTy dc)
  builderAppend $ text "  dcRepArgTys: " ><  ppr (dataConRepArgTys dc)
cvtDataConToMLIR dc = builderAppend $ text"// ***NOT HASKELL 98!***"



cvtTyConToMLIR :: TyCon -> Builder()
cvtTyConToMLIR c = do
 builderAppend $ text"//==TYCON: " >< ppr (tyConName c) >< (text "==")
 builderAppend $ text "//unique:">< ppr (tyConUnique c)
 builderAppend $ text"//|data constructors|"
 forM_  (tyConDataCons c)  cvtDataConToMLIR
 builderAppend $ text"//----"
 builderAppend $ text"//ctype: " >< ppr (tyConCType_maybe c)
 builderAppend $ text"//arity: " >< ppr (tyConArity c)
 builderAppend $ text"//binders: " >< ppr (tyConBinders c)

cvtModuleToMLIR :: DynFlags -> String -> ModGuts -> SDoc
cvtModuleToMLIR dfags phase guts = runBuilder $ do
  let doc_name = pprModuleName $ Module.moduleName $ mg_module guts 
  builderAppend $ comment doc_name
  builderAppend $ comment (text phase)
  builderAppend $ (text "module") <+> (text "{")
  -- builderAppend $ nest 4 $ mlirPrelude
  -- | TODO: convert to traverse?
  let vars = mconcat [getBindLHSs bind | bind <- (mg_binds guts)]
  forM_ (mg_tcs guts) cvtTyConToMLIR
  builderBracketRecursiveVars vars $  forM_ (filter shouldKeepBind (mg_binds guts)) (\b ->  builderNest 2 $ cvtTopBind b)
  builderAppend $ text "}"
  builderAppend $ text $ "// ============ Haskell Core ========================"
  builderAppend $ text $ dumpProgramAsCore dfags guts
  return ()

cvtBindRhs :: PossibleRecursiveVar -> CoreExpr -> Builder ()
cvtBindRhs name rhs = do
  rhs_name <- flattenExpr rhs
  builderAppend $ (text "hask.return(") >< rhs_name >< (text ")") 
  return ()
  -- let (_, rhs_name, rhs_preamble) = runBuilder_ (flattenExpr rhs) 0
  --     body = rhs_preamble $+$ ((text "hask.return(") >< rhs_name >< (text ")"))  
  -- in body


-- instance Outputable Var where
--   ppr var = sdocWithDynFlags $ \dflags ->
--             getPprStyle $ \ppr_style ->
--             if |  debugStyle ppr_style && (not (gopt Opt_SuppressVarKinds dflags))
--                  -> parens (ppr (varName var) <+> ppr_debug var ppr_style <+>
--                           dcolon <+> pprKind (tyVarKind var))
--                |  otherwise
--                  -> ppr (varName var) <> ppr_debug var ppr_style
--                  
-- 
--  ppr_debug :: Var -> PprStyle -> SDoc
-- ppr_debug (TyVar {}) sty
--   | debugStyle sty = brackets (text "tv")
-- ppr_debug (TcTyVar {tc_tv_details = d}) sty
--   | dumpStyle sty || debugStyle sty = brackets (pprTcTyVarDetails d)
-- ppr_debug (Id { idScope = s, id_details = d }) sty
--   | debugStyle sty = brackets (ppr_id_scope s <> pprIdDetails d)
-- ppr_debug _ _ = empty


-- | It turns out that varName knows when it should append the unique and when
-- | it should not.

-- pprName :: Name -> SDoc
-- pprName  (Name {n_sort = sort, n_uniq = uniq, n_occ = occ})
--   = getPprStyle $ \ sty ->
--     case sort of
--       WiredIn mod _ builtin   -> pprExternal sty uniq mod occ True  builtin
--       External mod            -> pprExternal sty uniq mod occ False UserSyntax
--       System                  -> pprSystem sty uniq occ
--       Internal                -> pprInternal sty uniq occ
-- -- | Print the string of Name unqualifiedly directly.
-- pprNameUnqualified :: Name -> SDoc
-- pprNameUnqualified Name { n_occ = occ } = ppr_occ_name occ
-- pprExternal :: PprStyle -> Unique -> Module -> OccName -> Bool -> BuiltInSyntax -> SDoc
-- pprExternal sty uniq mod occ is_wired is_builtin
--   | codeStyle sty = ppr mod <> char '_' <> ppr_z_occ_name occ
--         -- In code style, always qualify
--         -- ToDo: maybe we could print all wired-in things unqualified
--         --       in code style, to reduce symbol table bloat?
--   | debugStyle sty = pp_mod <> ppr_occ_name occ
--                      <> braces (hsep [if is_wired then text "(w)" else empty,
--                                       pprNameSpaceBrief (occNameSpace occ),
--                                       pprUnique uniq])
--   | BuiltInSyntax <- is_builtin = ppr_occ_name occ  -- Never qualify builtin syntax
--   | otherwise                   =
--         if isHoleModule mod
--             then case qualName sty mod occ of
--                     NameUnqual -> ppr_occ_name occ
--                     _ -> braces (ppr (moduleName mod) <> dot <> ppr_occ_name occ)
--             else pprModulePrefix sty mod occ <> ppr_occ_name occ
--   where
--     pp_mod = sdocWithDynFlags $ \dflags ->
--              if gopt Opt_SuppressModulePrefixes dflags
--              then empty
--              else ppr mod <> dot
-- pprInternal :: PprStyle -> Unique -> OccName -> SDoc
-- pprInternal sty uniq occ
--   | codeStyle sty  = pprUniqueAlways uniq
--   | debugStyle sty = ppr_occ_name occ <> braces (hsep [pprNameSpaceBrief (occNameSpace occ),
--                                                        pprUnique uniq])
--   | dumpStyle sty  = ppr_occ_name occ <> ppr_underscore_unique uniq
--                         -- For debug dumps, we're not necessarily dumping
--                         -- tidied code, so we need to print the uniques.
--   | otherwise      = ppr_occ_name occ   -- User style


-- | use the ppr of Var because it knows whether to print the unique ID or not.
-- | This function also makes sure to generate `@fib` for toplevel recursive
-- | binders. This will also work when we have `let`s [hopefully...]
-- | TODO: check what scope the variable `v` is defined in, this is a 
--         completely broken solution
cvtVar :: Var -> Builder SDoc
cvtVar v = do
  let name = unpackFS $ occNameFS $ getOccName v
  if name == "-#" then return (text "@\"-#\"")
  else if name == "+#" then return (text "@\"+#\"")
  else if name == "()" then return (text "@\"()\"")
  else do 
      isrec <- builderIsVarRecursive v
      if isrec then return (text "@" >< ppr v) else return (text "%" >< ppr v)

cvtVarORIGINAL_VERSION :: Var -> SDoc
cvtVarORIGINAL_VERSION v = 
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




cvtTopBindImpl :: (Var, CoreExpr) -> Builder ()
cvtTopBindImpl (b, e) = do 
    var_bind <- cvtVar b
    builderAppend $ text "hask.func" <+> var_bind <+> (text "{") 
    cvtBindRhs b e
    builderAppend $ text "}"
    return ()

cvtTopBind :: CoreBind -> Builder ()
cvtTopBind (NonRec b e) = cvtTopBindImpl (b, e)
    -- ((cvtVar b) <+> (text "=")) $$ 
    -- (nest 2 $ (text "hask.toplevel_binding") <+> (text "{") $$ (nest 2 $  (cvtBindRhs b e)) $$ (text "}"))
cvtTopBind (Rec bs) =
      forM_ bs cvtTopBindImpl
      -- (vcat $ [hsep [cvtVar b, text "=",
      --          braces_scoped (text "hask.toplevel_binding") (cvtBindRhs b e)] | (b, e) <- bs])


parenthesize :: SDoc -> SDoc
parenthesize sdoc = lparen >< sdoc >< rparen


cvtLit :: Literal -> SDoc
cvtLit l =
    case l of
#if MIN_VERSION_ghc(8,8,0)
      Literal.LitChar x ->  ppr x
      Literal.LitString x -> pprHsBytes x
      Literal.LitNullAddr -> error $ "unknown: how to handle null addr cvtLit(Literal.LitNullAddr)?"
      Literal.LitFloat x -> error "x" -- ppr x
      Literal.LitDouble x -> rational x
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
          Literal.LitNumInt -> text "hask.make_i64(" >< ppr n >< text ")"
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
      Literal.LitString x -> error "x" -- ppr x
      Literal.LitNullAddr -> error $ "unknown: how to handle null addr cvtLit(Literal.LitNullAddr)?"
      Literal.LitFloat x -> error "x" -- ppr x
      Literal.LitDouble x -> error "x" -- ppr x
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
cvtAltLhs (DataAlt altcon) = text "@" >< docDoubleQuote >< ppr (dataConName altcon) >< docDoubleQuote
cvtAltLhs (LitAlt l)       = cvtAltLHSLit l
cvtAltLhs DEFAULT          = text "\"default\""

arrow :: SDoc; arrow = text "->"

-- | wildcard is newtyped 
newtype Wild = Wild Var

-- | when we print a wild, we make sure it's unique.
cvtWild :: Wild -> Builder SDoc
cvtWild (Wild v) = cvtVar v
  -- let  varToUniqueName :: Var -> String
  --     varToUniqueName v = (unpackFS $ occNameFS $ getOccName v) ++ "_" ++ (show $ getUnique v)
  -- in text ("%" ++ varToUniqueName v)



cvtAltRHS :: Wild -> [Var] -> CoreExpr -> Builder ()
cvtAltRHS wild bnds rhs = do
    doc_wild <- cvtWild wild
    doc_binds <- traverse cvtVar bnds
    let params = hsep $ punctuate comma $ (doc_wild >< text ": !hask.untyped"):[b >< text ": !hask.untyped" | b <- doc_binds] 
    builderAppend $ text  "{"
    builderAppend $ text "^entry(" >< params >< text "):"
    -- | TODO: we need a way to nest stuff
    name_rhs <- builderNest 2 $ flattenExpr rhs
    builderAppend $ (text "hask.return(") >< name_rhs >< (text ")")
    builderAppend $ text "}"
    return ()

    -- let inner = (text "^entry(" >< params >< text "):") $$ (nest 2 $ preamble_rhs $$ ((text "hask.return(") >< name_rhs >< (text ")"))) 
    -- builderAppend (text "{" $$ (nest 2 inner) $$ text "}")

    
 -- -- | HACK: we need to start from something other than 100...
 -- let (i1, name_rhs, preamble_rhs) = runBuilder_ (flattenExpr rhs) i0
 --     params = hsep $ punctuate comma $ (cvtWild wild >< text ": !hask.untyped"):[cvtVar b >< text ": !hask.untyped" | b <- bnds]
 --     inner = (text "^entry(" >< params >< text "):") $$ (nest 2 $ preamble_rhs $$ ((text "hask.return(") >< name_rhs >< (text ")")))
 -- in (i1, (), (text "{") $$ (nest 2 inner) $$ (text "}"))


-- syntax: '['lhs '->' rhs ']'
-- TODO: add combinators to make this prettier. Something like 'builderSurround...'
cvtAlt :: Wild -> CoreAlt -> Builder ()
cvtAlt wild (lhs, bs, e) = do
 builderAppend $ lbrack >< cvtAltLhs lhs <+> Outputable.arrow
 cvtAltRHS wild bs e
 builderAppend $ rbrack
 return ()



isCaseAltsOnlyDefault :: [CoreAlt] -> Maybe CoreExpr
isCaseAltsOnlyDefault [(DEFAULT, _, e)] = Just e
isCaseAltsOnlyDefault _ = Nothing


-- | codegen an expression and give it a name?


-- | case needs scrutinee, alts, 
createCase :: MLIR.SSAId -> [(MLIR.SymbolRefId, MLIR.Region)] -> MLIR.Operation
createCase scrutinee alts = error "foo"


-- | create unique int
builderMakeUniqueInt :: State Int Int
builderMakeUniqueInt = do
  i <- get
  put (i + 1)
  return i

-- | create unique ID
builderMakeUniqueSSAId :: State Int MLIR.SSAId
builderMakeUniqueSSAId = do
  i <- get
  put (i + 1)
  return (MLIR.SSAId (show i))

codegenAltLHSLit :: Literal -> MLIR.AttributeValue
codegenAltLHSLit l =
    case l of
      -- Literal.LitNumber numty n _ ->
      --   case numty of
      --     Literal.LitNumInt -> MLIR.AttributeInteger n
      --     Literal.LitNumInt64 -> ppr n
      --     Literal.LitNumWord -> ppr n
      --     Literal.LitNumWord64 -> ppr n
      --     Literal.LitNumInteger -> ppr n
      --     Literal.LitNumNatural -> ppr n
      Literal.LitString s    -> MLIR.AttributeString (BS.unpack s)
      Literal.LitNumber _ i _ ->  MLIR.AttributeInteger i
      -- Literal.LitChar x ->  ppr x
      -- Literal.LitString x -> error "x" -- ppr x
      -- Literal.LitNullAddr -> error $ "unknown: how to handle null addr cvtLit(Literal.LitNullAddr)?"
      -- Literal.LitFloat x -> error "x" -- ppr x
      -- Literal.LitDouble x -> error "x" -- ppr x
      -- Literal.LitLabel x _ _ -> ppr $ unpackFS  x
      -- Literal.LitRubbish ->  error $ "unknown: how to handle null addr cvtLit(Literal.LitRubbish)?"
      -- Literal.LitInteger x _ -> ppr x

codegenAltLhs :: CoreSyn.AltCon -> MLIR.AttributeValue
codegenAltLhs (DataAlt altcon) =  MLIR.AttributeSymbolRef (MLIR.SymbolRefId (nameStableString (dataConName altcon)))
codegenAltLhs (LitAlt l)       = codegenAltLHSLit l
codegenAltLhs DEFAULT          = MLIR.AttributeSymbolRef (MLIR.SymbolRefId "default") -- text "\"default\""


-- cvtAltRHS :: Wild -> [Var] -> CoreExpr -> Builder ()
-- cvtAltRHS wild bnds rhs = do
--     doc_wild <- cvtWild wild
--     doc_binds <- traverse cvtVar bnds
--     let params = hsep $ punctuate comma $ (doc_wild >< text ": !hask.untyped"):[b >< text ": !hask.untyped" | b <- doc_binds] 
--     builderAppend $ text  "{"
--     builderAppend $ text "^entry(" >< params >< text "):"
--     -- | TODO: we need a way to nest stuff
--     name_rhs <- builderNest 2 $ flattenExpr rhs
--     builderAppend $ (text "hask.return(") >< name_rhs >< (text ")")
--     builderAppend $ text "}"
--     return ()

-- type Alt b = (AltCon, [b], Expr b)
-- |
-- | data AltCon
-- |   = DataAlt DataCon   --  ^ A plain data constructor: @case e of { Foo x -> ... }@.
-- |                       -- Invariant: the 'DataCon' is always from a @data@ type, and never from a @newtype@
-- |   | LitAlt  Literal   -- ^ A literal: @case e of { 1 -> ... }@
-- |                       -- Invariant: always an *unlifted* literal
-- |                       -- See Note [Literal alternatives]
-- |   | DEFAULT           -- ^ Trivial alternative: @case e of { _ -> ... }@
-- |    deriving (Eq, Data)

-- | int is the index, Var is the variable
-- return: attribute dict is LHS, region is RHS
codegenAlt' :: Wild -> (Int, CoreAlt) -> State Int (MLIR.AttributeDict, MLIR.Region)
codegenAlt' (Wild w) (i, (lhs, binds, rhs)) =  pure 
  (MLIR.AttributeDict [("alt" ++ show i, codegenAltLhs lhs)], error "foo")

codegenExpr' :: CoreExpr -> State Int ([MLIR.Operation], MLIR.SSAId)
codegenExpr' (Var x) = return ([], MLIR.SSAId (nameStableString . varName $ x))
codegenExpr' (Lam param body) = 
  error $ "unhandled: lambdas"
codegenExpr' (Case scr wild _ alts) = do
 (ops_scr, name_scr) <- codegenExpr' scr
 outs <- traverse (codegenAlt' (Wild wild))   ((zip [0,1..] alts) :: [(Int, CoreAlt)])
 curid <- builderMakeUniqueSSAId
 return ([], curid)


   


flattenExpr :: CoreExpr -> Builder SSAName
flattenExpr expr = 
      case expr of
        Var x -> (cvtVar x)--return ((text"%") >< ppr x)
        Lam param body -> do
            i <- builderMakeUnique
            let name_lambda = text $ "%lambda_" ++ show i
            doc_param <- cvtVar param
            builderAppend $ name_lambda <+> (text "=")  <+> text "hask.lambda(" >< doc_param >< (text ")") <+> (text "{")
            return_body <- builderNest 2 $ flattenExpr body
            builderAppend $ nest 2 $ (text "hask.return(") >< return_body >< (text ")")
            builderAppend $ (text "}")
            return name_lambda
        Case scrutinee wild _ as -> do
            name_scrutinee <- flattenExpr scrutinee
            i <- builderMakeUnique
            let name_case = text $ "%case_" ++ show i
            case isCaseAltsOnlyDefault as of
              Nothing -> do
                  builderAppend $ name_case <+> text "=" <+> text "hask.case " <+> name_scrutinee
                  forM_ as (cvtAlt (Wild wild))
                  return name_case
              Just defaultExpr -> do
                  doc_wild <- cvtWild (Wild wild)
                  builderAppend $ doc_wild <+> text "=" <+> text "hask.force (" >< name_scrutinee >< text ")"
                  -- builderAppend $ doc_wild <+> text "=" <+> text "hask.copy (" >< name_scrutinee >< text ")"
                  name_rhs <- flattenExpr defaultExpr
                  return name_rhs
        App f x -> do
            name_f <- flattenExpr f
            name_x <- flattenExpr x
            i <- builderMakeUnique
            let name_app = text ("%app_" ++ show i) 
            builderAppend $  (name_app <+> (text "=") <+> (text "hask.ap(") >< name_f >< comma <+> name_x >< (text ")")) 
            return name_app
        Lit l -> do
            i <- builderMakeUnique
            let name_lit = text $ "%lit_" ++ show i
            builderAppend $ name_lit <+> (text "=") <+> cvtLit l
            return name_lit  
        Type t -> do
          i <- builderMakeUnique
          let type_lit = text $ "%type_" ++ show i
          builderAppend $ type_lit <+> (text "=") <+> (text "hask.make_string(\"TYPEINFO_ERASED\")")
          return type_lit
        Tick _ e -> return (text ("TICK"))
        Cast _ e -> return (text ("CAST"))
        Let (NonRec b e) body -> do
            i <- builderMakeUnique 
            let name_unimpl = text ("%unimpl_let_nonrec" ++ show i)
            builderAppend $ name_unimpl <+> (text " = ") <+> (text "hask.make_i32(42)")
            return name_unimpl
          
        Let (Rec bs) body -> do 
            i <- builderMakeUnique 
            let name_unimpl = text ("%unimpl_let_rec" ++ show i)
            builderAppend $ name_unimpl <+> (text " = ") <+> (text "hask.make_i32(42)")
            return name_unimpl
