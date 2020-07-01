module Core2MLIR.MLIR where


-- // Identifiers
-- bare-id ::= (letter|[_]) (letter|digit|[_$.])*
data BareId = BareId String
-- bare-id-list ::= bare-id (`,` bare-id)*
-- ssa-id ::= `%` suffix-id
-- suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))
data SSAId = SSAId String
-- symbol-ref-id ::= `@` (suffix-id | string-literal)
data SymbolRefId = SymbolRefId String

