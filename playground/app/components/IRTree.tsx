import { useState } from "react";
import type { IRTreeNode } from "~/lib/wasm";

interface NodeProps {
  node: IRTreeNode;
}

function TreeNode({ node }: NodeProps) {
  const [expanded, setExpanded] = useState(true);
  const hasChildren = !!node.children?.length;

  return (
    <div className="tree-node">
      <div
        className="tree-node-label"
        onClick={(e) => {
          e.stopPropagation();
          if (hasChildren) setExpanded((v) => !v);
        }}
      >
        <span className={`tree-toggle ${hasChildren ? (expanded ? "expanded" : "collapsed") : "leaf"}`} />
        <span className="tree-text">{node.name}</span>
      </div>
      {hasChildren && (
        <div className={`tree-children ${expanded ? "" : "collapsed"}`}>
          {node.children!.map((child, i) => <TreeNode key={i} node={child} />)}
        </div>
      )}
    </div>
  );
}

interface IRTreeProps {
  nodes: IRTreeNode[] | undefined;
}

export function IRTree({ nodes }: IRTreeProps) {
  if (!nodes || nodes.length === 0) {
    return <div className="tree-container">(empty)</div>;
  }
  return (
    <div className="tree-container">
      {nodes.map((node, i) => <TreeNode key={i} node={node} />)}
    </div>
  );
}
