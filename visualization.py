"""
Visualization utilities for TreatmentEffectEstimator.

Requires: graphviz (Python package) + Graphviz system install for PDF rendering.
Optional: IPython for inline display in notebooks.
"""

try:
    import graphviz
except ImportError as e:
    graphviz = None
    _graphviz_import_error = e

try:
    from IPython.display import display
except Exception:
    display = None


class TreatmentEffectVisualizer:
    def __init__(self, estimator):
        if graphviz is None:
            raise ImportError(
                "graphviz is required for visualization. Install with:\n"
                "  pip install graphviz\n"
                "And install the Graphviz system package so 'dot' is on your PATH."
            ) from _graphviz_import_error

        self.estimator = estimator
        self.graph = graphviz.Digraph(
            format="pdf",
            node_attr={"shape": "plaintext"},
            graph_attr={"rankdir": "TB", "ranksep": "0.2", "nodesep": "0.1"},
        )

    def visualize(self):
        for treatment_name, root_node in self.estimator.roots.items():
            self._visualize_node(root_node, treatment_name)
        return self.graph

    def _visualize_node(self, node, treatment_name, parent_id=None):
        if node is None:
            return

        node_id = f"node_{id(node)}"

        # Leaf node label: show estimate + std if available
        if node.left is None and node.right is None:
            tau = getattr(node, "tau", None)
            std = getattr(node, "std", None)

            tau_str = f"{tau:.2f}" if tau is not None else "NA"
            std_str = f"{std:.1f}" if std is not None else "NA"

            label = f"""<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                <TR><TD><b>{tau_str}</b></TD></TR>
                <TR><TD>({std_str})</TD></TR>
                <TR><TD>Node {node.node_name}</TD></TR>
                <TR><TD>Samples: {node.n}</TD></TR>
            </TABLE>>"""
        else:
            label_lines = [f"Node {node.node_name}", f"Samples: {node.n}"]
            if node.split is not None:
                covariate, split_value = node.split
                if covariate == "Year":
                    split_value = split_value + 2016
                label_lines.append(f"{covariate} ≤ {split_value:.3f}")
            label = "\\n".join(label_lines)

        self.graph.node(node_id, label=label)

        if parent_id is not None:
            self.graph.edge(parent_id, node_id)

        self._visualize_node(node.left, treatment_name, node_id)
        self._visualize_node(node.right, treatment_name, node_id)


def visualize_treatment_effect(estimator, filename="tree"):
    """
    Build and render the treatment effect trees.

    filename: base filename without extension (recommended), e.g. "tree"
    Returns the graphviz.Digraph object.
    """
    visualizer = TreatmentEffectVisualizer(estimator)
    tree_graph = visualizer.visualize()

    if display is not None:
        display(tree_graph)

    output_path = tree_graph.render(filename=filename, cleanup=True)
    print(f"Tree visualization saved as: {output_path}")
    return tree_graph
