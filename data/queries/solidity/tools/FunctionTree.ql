/**
 * @name FunctionTree
 * @description Extracts all functions and modifiers from Solidity contracts
 * @kind problem
 * @id solidity/function-tree
 */

import codeql.solidity.ast.internal.TreeSitter

/**
 * Represents callable elements (functions and modifiers) in Solidity
 */
class SolidityCallable extends Solidity::AstNode {
  string name;
  
  SolidityCallable() {
    (
      this instanceof Solidity::FunctionDefinition and
      name = this.(Solidity::FunctionDefinition).getName().getValue()
    )
    or
    (
      this instanceof Solidity::ModifierDefinition and
      name = this.(Solidity::ModifierDefinition).getName().getValue()
    )
  }
  
  string getName() { result = name }
}

from SolidityCallable callable
select 
  callable.getName() as function_name,
  callable.getLocation().getFile().getAbsolutePath() as file,
  callable.getLocation().getStartLine() as start_line,
  callable.getLocation().getFile().getAbsolutePath() + ":" + callable.getLocation().getStartLine().toString() as function_id,
  callable.getLocation().getEndLine() as end_line,
  "" as caller_id
