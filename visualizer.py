"""Draw the compiled graph to a PNG.

Lifted from the course lessons, where the same helper sits next to every
LangGraph example. Kept because the picture is the quickest way to show that
create_agent really does compile to a graph with a loop in it, rather than to a
while statement somewhere.

The draw needs network access (mermaid.ink renders it), so a failure here must
never take the agent down with it - the run is the deliverable, the picture is a
convenience.
"""

import io


def visualize(graph, output_file_name: str) -> bool:
    """Save the graph as a PNG. Returns whether it worked."""
    try:
        from PIL import Image

        png = graph.get_graph().draw_mermaid_png()
        Image.open(io.BytesIO(png)).save(output_file_name, "PNG")
        return True
    except Exception:
        # Silent on purpose. It is a picture.
        return False


def mermaid(graph) -> str:
    """The graph as mermaid source, which needs no network at all."""
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as problem:
        return f"(could not draw the graph: {problem})"
