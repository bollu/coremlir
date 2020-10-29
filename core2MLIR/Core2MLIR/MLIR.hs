-- | https://github.com/llvm/llvm-project/blob/master/mlir/docs/LangRef.md
module Core2MLIR.MLIR where
import qualified Data.List.NonEmpty as NE
import Outputable as O


-- // Identifiers
-- bare-id ::= (letter|[_]) (letter|digit|[_$.])*
newtype BareId = BareId String

instance Outputable BareId where
  ppr (BareId x) = ppr x

-- bare-id-list ::= bare-id (`,` bare-id)*
-- ssa-id ::= `%` suffix-id
-- suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))
data SSAId = SSAId String

instance Outputable SSAId where
  ppr (SSAId x) = ppr ('%':x)

-- symbol-ref-id ::= `@` (suffix-id | string-literal)
data SymbolRefId = SymbolRefId String

instance Outputable SymbolRefId where
  ppr (SymbolRefId x) = ppr ('@':x)
-- operation
-- region ::= `{` block* `}`
newtype Region = Region [Block]

instance Outputable Region where
  ppr (Region bs) = lbrace <+> nest 4 (vcat (map ppr bs)) <+> rbrace
 
-- region-list       ::= region (`,` region)*
newtype RegionList = RegionList [Region]
-- block           ::= block-label operation+
data Block = Block BlockLabel (NE.NonEmpty Operation)

instance Outputable Block where
  
-- block-label     ::= block-id block-arg-list? `:`
data BlockLabel = BlockLabel BlockId BlockArgList

instance Outputable BlockLabel where
  ppr (BlockLabel name args) = 
    let prettyArgs args = parens (hcat $ punctuate comma [ppr v O.<> colon O.<> ppr t | (v, t) <- args])
    in ppr name  <+> prettyArgs args O.<> colon
-- // Non-empty list of names and types.
-- value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*
-- block-arg-list ::= `(` value-id-and-type-list? `)`
-- value-id-and-type ::= value-id `:` type
-- // Non-empty list of names and types.
-- value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*
type BlockArgList = [(SSAId, Type)]-- [(ValueId, Type)]
-- block-id        ::= caret-id
-- caret-id        ::= `^` suffix-id
newtype BlockId = BlockId String -- BlockId SuffixId 
instance Outputable BlockId where
  ppr (BlockId x) = ppr ('^':x)
-- suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))
newtype SuffixId = SuffixId String


-- attribute-dict ::= `{` `}`
--                  | `{` attribute-entry (`,` attribute-entry)* `}`
-- attribute-entry ::= dialect-attribute-entry | dependent-attribute-entry
-- dialect-attribute-entry ::= dialect-namespace `.` bare-id `=` attribute-value
-- dependent-attribute-entry ::= dependent-attribute-name `=` attribute-value
-- dependent-attribute-name ::= ((letter|[_]) (letter|digit|[_$])*)
--                            | string-literal
type AttributeName = String
data AttributeDict = AttributeDict [(AttributeName, AttributeValue)]

-- attribute-value ::= attribute-alias | dialect-attribute | standard-attribute
-- standard-attribute ::=   affine-map-attribute
--                        | array-attribute
--                        | bool-attribute
--                        | dictionary-attribute
--                        | elements-attribute
--                        | float-attribute
--                        | integer-attribute
--                        | integer-set-attribute
--                        | string-attribute
--                        | symbol-ref-attribute
--                        | type-attribute
--                        | unit-attribute
-- 
data AttributeValue = AttributeSymbolRef SymbolRefId | AttributeString String | AttributeInteger Integer | AttributeType Type 


-- operation         ::= op-result-list? (generic-operation | custom-operation)
--                       trailing-location?
-- generic-operation ::= string-literal `(` value-use-list? `)`  successor-list?
--                       (`(` region-list `)`)? attribute-dict? `:` function-type
data Operation = 
  Operation { opname :: String, 
                     opvals :: ValueUseList, 
                     opsuccs :: SuccessorList, 
                     opregions :: RegionList,
                     opattrs :: AttributeDict,
                     opty :: FunctionType
                    }


-- // MLIR functions can return multiple values.
-- function-result-type ::= type-list-parens
--                        | non-function-type
-- 
-- function-type ::= type-list-parens `->` function-result-type
type FunctionType = ([Type], Type)
-- type ::= type-alias | dialect-type | standard-type
-- standard-type ::=     complex-type
--                     | float-type
--                     | function-type
--                     | index-type
--                     | integer-type
--                     | memref-type
--                     | none-type
--                     | tensor-type
--                     | tuple-type
--                     | vector-type
-- dialect-type ::= '!' opaque-dialect-item
-- opaque-dialect-item ::= dialect-namespace '<' string-literal '>'
-- signed-integer-type ::= `si` [1-9][0-9]*
-- unsigned-integer-type ::= `ui` [1-9][0-9]*
-- signless-integer-type ::= `i` [1-9][0-9]*
-- integer-type ::= signed-integer-type |
--                  unsigned-integer-type |
--                  signless-integer-type
-- 
data Type = TypeDialect DialectNamespace String
    | TypeIntegerSignless Int  -- ^ width
instance Outputable Type where
  ppr (TypeDialect ns x) = ppr ns  O.<> angleBrackets (ppr x)
  ppr (TypeIntegerSignless i) = ppr 'i' O.<> ppr i

data SuccessorList =SuccessorList
-- custom-operation  ::= bare-id custom-operation-format
-- op-result-list    ::= op-result (`,` op-result)* `=`
newtype OpResultList = NonEmpty OpResult
-- op-result         ::= value-id (`:` integer-literal)
newtype OpResult = OpResult String -- TODO: add the maybe int to pick certain results out
-- successor-list    ::= successor (`,` successor)*
-- successor         ::= caret-id (`:` bb-arg-list)?
-- region-list       ::= region (`,` region)*
-- trailing-location ::= (`loc` `(` location `)`)?
-- // Uses of value, e.g. in an operand list to an operation.
-- value-use ::= value-id
-- value-use-list ::= value-use (`,` value-use)*
newtype ValueUseList = ValueUseList [SSAId] -- [ValueId]

-- value-id ::= `%` suffix-id
-- newtype ValueId = ValueId String


-- // This is a common way to refer to a value with a specified type.
-- ssa-use-and-type ::= ssa-use `:` type
-- 
-- // Non-empty list of names and types.
-- ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*

-- dialect-namespace ::= bare-id
newtype DialectNamespace = DialectNamespace String
instance Outputable DialectNamespace where 
  ppr (DialectNamespace x) = ppr x
-- opaque-dialect-item ::= dialect-namespace '<' string-literal '>'
-- pretty-dialect-item ::= dialect-namespace '.' pretty-dialect-item-lead-ident
--                                               pretty-dialect-item-body?
-- pretty-dialect-item-lead-ident ::= '[A-Za-z][A-Za-z0-9._]*'
-- pretty-dialect-item-body ::= '<' pretty-dialect-item-contents+ '>'
-- pretty-dialect-item-contents ::= pretty-dialect-item-body
--                               | '(' pretty-dialect-item-contents+ ')'
--                               | '[' pretty-dialect-item-contents+ ']'
--                               | '{' pretty-dialect-item-contents+ '}'
--                               | '[^[<({>\])}\0]+'
-- 
-- dialect-type ::= '!' opaque-dialect-item
-- dialect-type ::= '!' pretty-dialect-item
data DialectType = DialectType  DialectNamespace String
