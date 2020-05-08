#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
MIT License

Copyright (c) 2018 Adolfo Valdivieso

"""

import keras
import os
import argparse
import os.path as osp
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference


def create_output_folder(output_folder):
    # create output directory
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    return output_folder


def load_keras_model(model_json_file_path, weights_file_path):
    # load keras model
    with open(model_json_file_path, 'r') as json_file:
        loaded_model_json = json_file.read() # read json file
        net_model = keras.models.model_from_json(loaded_model_json) # create model from json
        net_model.load_weights(weights_file_path) # load weigths to model
        return net_model
    return None


def generate_nodes_names(basename, num_layer_nodes):
    return [basename + "_" + str(i) for i in range(num_layer_nodes)]


def set_output_node_names(net_model, node_names):
    for i, node_name in enumerate(node_names):
        net_model.outputs[i] = tf.identity(net_model.outputs[i], name=node_name) # setup node name
    return net_model


def set_model_output_node_names(net_model, output_nodes_name):
    output_nodes_names = generate_nodes_names(output_nodes_name, len(net_model.outputs))
    new_net_model = set_output_node_names(net_model, output_nodes_names)
    return new_net_model, output_nodes_names


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--input_model",
        type=str,
        required=False,
        default="model.json",
        help="Keras model as a json file that we want to trasnform into a tensorflow buffer and optimize for Opencv Dnn.")
    parser.add_argument(
        "--input_weigths",
        type=str,
        required=False,
        default="model.h5",
        help="Keras weigths file (.h5) that contains the weigths of out Keras model.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exported_model/",
        required=False,
        help="Directory where you want the output results to be saved.")
    parser.add_argument(
        "--output_name",
        type=str,
        default="model.pb",
        help="Name for the resulting tensorflow buffer (.pb).")
    parser.add_argument(
        "--output_nodes_name",
        type=str,
        default="output",
        help="Name that will be assign to the output nodes.")
    parser.add_argument(
        "--input_nodes_name",
        type=str,
        default="input",
        help="Name that will be assign to the input nodes.")
    return parser.parse_args()


if __name__ == '__main__':
    # parse args
    ARGS = parse_args()
    # Sets the learning phase to a fixed value.
    keras.backend.set_learning_phase(0)
    # load model from json file and load weigths
    net_model = load_keras_model(ARGS.input_model, ARGS.input_weigths)
    net_model.summary()
    assert net_model is not None, 'Model can not be loaded.'
    # set names for input and output nodes of the model
    new_net_model, output_nodes_names = set_model_output_node_names(net_model, ARGS.output_nodes_name)

    print ""
    print "MODEL "
    net_model.summary()
    print ""

    input_nodes_names = [node.name[:-2] for node in net_model.inputs]
    output_nodes_names = [node.name[:-2] for node in net_model.outputs]

    print ""
    print "Inputs:", input_nodes_names
    print "Outputs:", output_nodes_names
    print ""

    # get sesssion
    sess = keras.backend.get_session()

    output_folder = create_output_folder(ARGS.output_dir)

    saver = tf.train.Saver(tf.global_variables())

    # freeze graph: trasnform variable placeholders into constants
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_nodes_names)
    # save frozen graph
    frozen_graph_name = "frozen_"+ARGS.output_name
    graph_io.write_graph(constant_graph, output_folder, frozen_graph_name, as_text=False)
    print('Saved frozen graph at: ', osp.join(output_folder, frozen_graph_name))

    # Optimize for inference

    # load frozen graph
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(os.path.join(output_folder, frozen_graph_name), "r") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    # print node names
    print ""
    print "NODES"
    print "====="
    for node in input_graph_def.node:
        print node.name
    print ""

    # optimze graph
    output_graph_def = optimize_for_inference(
        input_graph_def,
        input_nodes_names, output_nodes_names, # an array of output nodes
        tf.float32.as_datatype_enum)

    # save graph to output file
    optimized_graph_name = "optimized_"+ARGS.output_name
    f = tf.gfile.FastGFile(os.path.join(output_folder, optimized_graph_name), "w")
    f.write(output_graph_def.SerializeToString())
    print('Saved optimized graph (ready for inference) at: ', osp.join(output_folder, optimized_graph_name))

    with tf.gfile.FastGFile(osp.join(output_folder, "optimized_model.pb"), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Remove Const nodes.
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].op == 'Const':
            del graph_def.node[i]
        for attr in ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim',
                     'use_cudnn_on_gpu', 'Index', 'Tperm', 'is_training',
                     'Tpaddings']:
            if attr in graph_def.node[i].attr:
                del graph_def.node[i].attr[attr]

    # Save as text.
    tf.train.write_graph(graph_def, "", osp.join(output_folder, "model.pbtxt"), as_text=True)
    print('Saved .pbtxt file')
