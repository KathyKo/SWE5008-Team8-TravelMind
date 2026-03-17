from main import build_travel_graph
import os

def visualize():
    print("Building travel graph...")
    graph = build_travel_graph()
    
    # 1. Generate ASCII diagram
    print("\n--- ASCII Workflow Diagram ---")
    try:
        # LangGraph 0.2+ way to get graph representation
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Could not generate ASCII diagram: {e}")

    # 2. Generate PNG via Mermaid.ink
    print("\nGenerating graph_workflow.png...")
    try:
        # Generate Mermaid Markdown
        mermaid_md = graph.get_graph().draw_mermaid()
        print("\n--- Mermaid Markdown (copy & paste to mermaid.live) ---")
        print(mermaid_md)
        
        # This will work if you have a connection to mermaid.ink or local graphviz
        png_data = graph.get_graph().draw_mermaid_png()
        with open("graph_workflow.png", "wb") as f:
            f.write(png_data)
        print("\nSuccessfully saved graph_workflow.png")
    except Exception as e:
        print(f"\nCould not generate PNG: {e}")
        print("Try installing pygraphviz or checking your internet connection for Mermaid.ink.")

if __name__ == "__main__":
    visualize()
