import codeql.solidity.ast.internal.TreeSitter

from Solidity::ModifierDefinition mod
where exists(mod.getName())
select 
  mod.getName().getValue(),
  mod.getLocation().getFile().getRelativePath(),
  mod.getLocation().getStartLine(),
  mod.toString(),
  mod.getLocation().getEndLine()
