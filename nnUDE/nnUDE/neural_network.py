import abc
from dataclasses import dataclass
from enum import Enum

from typing import List, Optional, Dict, Any, Callable
import petab
from petab.C import (
    PARAMETER_ID,
    NOMINAL_VALUE,
    LOWER_BOUND,
    UPPER_BOUND,
    PARAMETER_SCALE,
    LIN,
    LOG10,
    LOG,
    ESTIMATE,
    INITIALIZATION_PRIOR_TYPE,
    INITIALIZATION_PRIOR_PARAMETERS,
    NORMAL,
)


@dataclass
class Node:
    id_: str
    index: int
    input_nodes: Optional[List["Node"]]
    activation_function: Optional[Callable[[str], str]]
    formula: Optional[str] = None
    """The formula that describes how this node is computed with respect to the input species IDs."""


@dataclass
class Layer:
    id_: str
    index: Optional[int]
    nodes: List[Node]


@dataclass
class NeuralNetwork:
    id_: str
    input_layer: Layer
    output_species_ids: List[str]
    layers: List[Layer]
    #activation_function: str
    regularization: Dict[str, Any]

    def __post_init__(self):
        dimension_last_layer = len(self.layers[-1].nodes)
        n_output_species_ids = len(self.output_species_ids)
        if dimension_last_layer != n_output_species_ids:
            raise ValueError(
                "The dimension of the last layer must match the dimension of `output_species_ids`. "
                f"Dimension of last layer: {dimension_last_layer}. "
                f"Dimension of `output_species_ids`: {n_output_species_ids}."
            )


def input_layer_from_species_ids(
    neural_network_id: str,
    species_ids: List[str],
) -> List[Node]:
    nodes = []
    for species_index, species_id in enumerate(species_ids):
        node = Node(
            id_=species_id,
            index=species_index,
            input_nodes=None,
            activation_function=None,
            formula=species_id,
        )
        nodes.append(node)
    layer = Layer(
        id_=f"neural_network_{neural_network_id}__layer_input",
        index=None,
        nodes=nodes,
    )
    return layer


def create_feedforward_network(
    neural_network_id: str,
    dimensions: List[int],
    activation_function: str,
    input_species_ids: Layer,
    output_species_ids: List[str],
    identity_last_layer: bool = False,
):
    layers = []
    input_layer = input_layer_from_species_ids(
        neural_network_id=neural_network_id,
        species_ids=input_species_ids,
    )

    layer0 = input_layer
    for layer_index, dimension in enumerate(dimensions):
        nodes = []
        layer_id = f"neural_network_{neural_network_id}__layer_{layer_index}"

        activation_function_layer = activation_function
        if identity_last_layer and (layer_index == len(dimensions) - 1):
            activation_function_layer = ActivationFunction.identity

        for node_index in range(dimension):
            node = Node(
                id_=layer_id + f"__node__{node_index}",
                index=node_index,
                input_nodes=layer0.nodes,
                activation_function=activation_function_layer,
            )
            nodes.append(node)

        layer = Layer(
            id_=layer_id,
            index=layer_index,
            nodes=nodes,
        )

        layers.append(layer)
        layer0 = layer

    neural_network = NeuralNetwork(
        id_=neural_network_id,
        input_layer=input_layer,
        layers=layers,
        output_species_ids=output_species_ids,
        regularization=None,
    )

    return neural_network


class ActivationFunction:
    def identity(formula0: str):
        return formula0

    def gaussian(formula0: str):
        """N.B.: not similar to an RBF network"""
        return f"exp(-({formula0})^2)"

    def relu(formula0: str):
        return f"piecewise({formula0}, {formula0} > 0, 0)"
    
    def tanh(formula0: str):
        return f"tanh({formula0})"
    
    def sigmoid(formula0: str):
        return f"1/(1 + exp(-({formula0})))"
    
    def softplus(formula0: str):
        return f"log(1 + exp({formula0}))"
    
    def swish(formula0: str):
        """Without the trainable beta parameter.
    
        i.e.: sigmoid linear unit
        """
        return f"({formula0}) * ({sigmoid_activation_function(formula0)})"

