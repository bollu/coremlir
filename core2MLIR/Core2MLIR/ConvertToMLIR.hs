{-# LANGUAGE CPP #-}
{-# LANGUAGE ViewPatterns #-}
module Core2MLIR.ConvertToMLIR(codegenModuleToMLIRSDoc) where
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


haskdialect :: MLIR.DialectNamespace;
haskdialect = MLIR.DialectNamespace "hask"

docDoubleQuote :: SDoc
docDoubleQuote = text "\""

-- | name of the expression
type SSAName = SDoc

-- | keep around stuff that can be recursive, so we can emit them 
-- differently.
type PossibleRecursiveVar = Var

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


                                                                
getBindLHSs :: CoreBind -> [Var]
getBindLHSs (NonRec b e) = [b]
getBindLHSs (Rec bs) = [ b | (b, _) <- bs]                                


codegenTyCon :: TyCon -> State Int MLIR.Operation
codegenTyCon c = error "codegen ty con" -- MLIR.Comment (tyConCType_maybe c) -- pure (MLIR.defaultop)


codegenModuleToMLIRSDoc ::DynFlags -> String -> ModGuts -> SDoc
codegenModuleToMLIRSDoc dflags phase guts = 
  vcat $ (map ppr (codegenModuleToMLIR dflags phase guts))

-- | kill me now I'm using { ... }
codegenModuleToMLIR :: DynFlags -> String -> ModGuts -> [MLIR.Operation]
codegenModuleToMLIR dfags phase guts = evalState (do
   -- tys <- forM (mg_tcs guts) codegenTyCon
   let tys = []
   topbindss <- forM (filter shouldKeepBind (mg_binds guts)) codegenTopBind
   return (tys ++ concat topbindss)) 0
  

codegenBindRhs :: PossibleRecursiveVar -> CoreExpr -> State Int MLIR.Region
codegenBindRhs name rhs = do
  return MLIR.defaultRegion



-- nameStableString . varName
varToString :: Var -> String
varToString v = nameStableString (varName v)
  -- (unpackFS $ occNameFS $ getOccName v) ++ "_" ++ (show $ getUnique v)
  -- (unpackFS $ occNameFS $ getOccName v)


-- | hask.return
haskreturnop :: (MLIR.SSAId, MLIR.Type) -> MLIR.Operation
haskreturnop (retv, rett) = MLIR.defaultop {
       MLIR.opname = "hask.return", 
       MLIR.opvals = MLIR.ValueUseList [retv],
       MLIR.opty = MLIR.FunctionType [rett] [rett] 
  }

haskvalty :: MLIR.Type
haskvalty = MLIR.TypeCustom (text "!hask.value")

codegenTopBindImpl :: (Var, CoreExpr) -> State Int MLIR.Operation
codegenTopBindImpl (varname, e) =  do
  (ops, finalval) <- codegenExpr' e
  let entry = MLIR.block "entry"  [] (ops ++ [haskreturnop (finalval, haskvalty)])
  let r = MLIR.Region [entry]
  return MLIR.defaultop { 
       MLIR.opname = "hask.func", 
       MLIR.opattrs = MLIR.AttributeDict [("sym_name", MLIR.AttributeString (varToString varname))],
       MLIR.opregions = MLIR.RegionList [r]
     }

codegenTopBind :: CoreBind -> State Int [MLIR.Operation]
codegenTopBind (NonRec b e) = do 
   bind <- codegenTopBindImpl (b, e)
   return [bind] 

codegenTopBind (Rec bs) = forM bs codegenTopBindImpl


arrow :: SDoc; arrow = text "->"

-- | wildcard is newtyped 
newtype Wild = Wild Var

isCaseAltsOnlyDefault :: [CoreAlt] -> Maybe CoreExpr
isCaseAltsOnlyDefault [(DEFAULT, _, e)] = Just e
isCaseAltsOnlyDefault _ = Nothing




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

codegenAltLHSLit' :: Literal -> MLIR.AttributeValue
codegenAltLHSLit' l =
    case l of
      Literal.LitString s    -> MLIR.AttributeString (BS.unpack s)
      Literal.LitNumber _ i _ ->  MLIR.AttributeInteger i
codegenAltLhs' :: CoreSyn.AltCon -> MLIR.AttributeValue
codegenAltLhs' (DataAlt altcon) =  
   MLIR.AttributeSymbolRef (MLIR.SymbolRefId (nameStableString (dataConName altcon)))
codegenAltLhs' (LitAlt l)       = codegenAltLHSLit' l
codegenAltLhs' DEFAULT          = MLIR.AttributeSymbolRef (MLIR.SymbolRefId "default") -- text "\"default\""


codegenAltRHS' :: Wild -> [Var] -> CoreExpr -> State Int MLIR.Region
codegenAltRHS' wild bnds rhs = do
 (ops, finalval) <- codegenExpr' rhs
 let entry = MLIR.block "entry"  
                    [(MLIR.SSAId (varToString b), haskvalty)| b <- bnds]
                    (ops ++ [haskreturnop (finalval, haskvalty)])
 let r = MLIR.Region [entry]
 -- doc_wild <- cvtWild wild
 -- doc_binds <- traverse cvtVar bnds
 -- let params = hsep $ punctuate comma $ (doc_wild >< text ": !hask.untyped"):[b >< text ": !hask.untyped" | b <- doc_binds]
 -- builderAppend $ text  "{"
 -- builderAppend $ text "^entry(" >< params >< text "):"
 -- -- | TODO: we need a way to nest stuff
 -- name_rhs <- builderNest 2 $ flattenExpr rhs
 -- builderAppend $ (text "hask.return(") >< name_rhs >< (text ")")
 -- builderAppend $ text "}"
 return r

-- | int is the index, Var is the variable
-- return: attribute dict is LHS, region is RHS
codegenAlt' :: Wild -> (Int, CoreAlt) -> State Int (MLIR.AttributeDict, MLIR.Region)
codegenAlt' (Wild w) (i, (lhs, binds, rhs)) =  do
  let attrs = MLIR.AttributeDict [("alt" ++ show i, codegenAltLhs' lhs)]
  r <-  codegenAltRHS' (Wild w) binds rhs
  pure (attrs, r)
codegenExpr' :: CoreExpr -> State Int ([MLIR.Operation], MLIR.SSAId)
codegenExpr' (Var x) = 
  return ([], MLIR.SSAId (varToString $ x))
codegenExpr' (Lam param body) = do
  (ops, finalval) <- codegenExpr' body
  -- TODO: add param
  let entry = MLIR.block "entry"  [(MLIR.SSAId (varToString param), haskvalty)] 
                    (ops ++ [haskreturnop (finalval, haskvalty)])
  let r = MLIR.Region [entry]
  curid <- builderMakeUniqueSSAId
  let op = MLIR.defaultop {
       MLIR.opname = "hask.lambda", 
       MLIR.opregions = MLIR.RegionList [r],
       MLIR.opresults = MLIR.OpResultList [curid],
       -- | interesting! can have closure captured variables as parameters =)
       MLIR.opty = MLIR.FunctionType []  [haskvalty]
  }
  return ([op], curid)

codegenExpr' (App f x) = do
 (fops, fname) <- codegenExpr' f
 (xops, xname) <- codegenExpr' x
 curid <- builderMakeUniqueSSAId
 let op = MLIR.defaultop {
       MLIR.opname = "hask.ap", 
       MLIR.opvals = MLIR.ValueUseList [fname, xname],
       MLIR.opresults = MLIR.OpResultList [curid],
       MLIR.opty = MLIR.FunctionType [haskvalty, haskvalty] [haskvalty]
 }
 return (fops ++ xops ++ [op], curid)

codegenExpr' (Case scr wild _ alts) = do
 (scrops, scrname) <- codegenExpr' scr
 attrRgns <- traverse (codegenAlt' (Wild wild))   ((zip [0,1..] alts) :: [(Int, CoreAlt)])
 curid <- builderMakeUniqueSSAId
 let op = MLIR.defaultop { 
     MLIR.opname = "hask.case",
     MLIR.opvals = MLIR.ValueUseList [scrname],
     MLIR.opregions = MLIR.RegionList (map snd attrRgns),
     MLIR.opresults = MLIR.OpResultList [curid],
     MLIR.opattrs = mconcat (map fst attrRgns),
     MLIR.opty = MLIR.FunctionType [haskvalty] [haskvalty]
  }
 return (scrops ++ [op], curid)

codegenExpr' (Let _ e) = do
 curid <- builderMakeUniqueSSAId
 let op = MLIR.defaultop { MLIR.opname = "hask.let", MLIR.opresults = MLIR.OpResultList [curid] }
 return ([op], curid)

codegenExpr' (Type t) = do
 curid <- builderMakeUniqueSSAId
 let op = MLIR.defaultop { 
            MLIR.opname = "hask.type", 
            MLIR.opresults = MLIR.OpResultList [curid],
            MLIR.opty = MLIR.FunctionType [] [haskvalty]
          }
 return ([op], curid)

codegenExpr' (Lit t) = do
 curid <- builderMakeUniqueSSAId
 let op = MLIR.defaultop { 
    MLIR.opname = "hask.lit", 
    -- | TODO: rename to codegenLit?
    MLIR.opattrs = MLIR.AttributeDict [("value", codegenAltLHSLit' t)],
    -- | FML. I might have to pick the type based on nonsense.
    MLIR.opty = MLIR.FunctionType []   [haskvalty],
    MLIR.opresults = MLIR.OpResultList [curid]
  }
 return ([op], curid)

codegenExpr' (Cast _ e) = do
 curid <- builderMakeUniqueSSAId
 let op = MLIR.defaultop { 
            MLIR.opname = "hask.cast",
            MLIR.opresults = MLIR.OpResultList [curid]
          }
 return ([op], curid)

codegenExpr' (Tick _ e) = do
 curid <- builderMakeUniqueSSAId
 let op = MLIR.defaultop { 
             MLIR.opname = "hask.tick",
             MLIR.opresults = MLIR.OpResultList [curid]
          }
 return ([op], curid)

codegenExpr' _ = do 
 curid <- builderMakeUniqueSSAId
 let op = MLIR.defaultop { 
            MLIR.opname = "unk",  
            MLIR.opresults = MLIR.OpResultList [curid]
          }
 return ([op], curid)
