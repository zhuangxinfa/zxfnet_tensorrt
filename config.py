import graphsurgeon as gs
import tensorflow as tf

Output = gs.create_node("dot_op_trt",op="my_matmul", dtype=tf.float32)

namespace_plugin_map = {
    "dot_op": Output
}

def preprocess(dynamic_graph):
    # Now create a new graph by collapsing namespaces
    dynamic_graph.collapse_namespaces(namespace_plugin_map)
