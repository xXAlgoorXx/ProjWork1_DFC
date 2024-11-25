import onnx
from onnx import helper, numpy_helper

# Load the ONNX model
model_path = "hailoDFC/models/clip_simpel.onnx"
model = onnx.load(model_path)

# Define the target node at which to cut the graph
target_node_name = "/attnpool/Add"  # Replace with the name of the node

# Identify all nodes up to and including the target node
nodes_to_keep = []
used_initializer_names = set()  # To track required initializers
for node in model.graph.node:
    nodes_to_keep.append(node)
    for input_name in node.input:
        used_initializer_names.add(input_name)  # Track weights being used
    if node.name == target_node_name:
        break  # Stop after including the target node

# Filter the initializers to retain only those used in the pruned graph
new_initializers = [
    initializer
    for initializer in model.graph.initializer
    if initializer.name in used_initializer_names
]

# Create a new graph with the nodes and initializers to keep
new_graph = helper.make_graph(
    nodes=nodes_to_keep,
    name=model.graph.name,
    inputs=model.graph.input,  # Retain original inputs
    outputs=[
        helper.make_tensor_value_info(
            target_node_name, onnx.TensorProto.FLOAT, None  # Update as per target node
        )
    ],
    initializer=new_initializers,  # Keep only required initializers
)

# Create and save the new model
new_model = helper.make_model(new_graph, producer_name="cut_graph_script")
new_model_path = "cut_model.onnx"
onnx.save(new_model, new_model_path)
print(f"Model has been cut and saved to {new_model_path}")
