import copy
import petab


from petab.C import (
    PARAMETER_ID,
    NOMINAL_VALUE,
    PARAMETER_SCALE,
    LOWER_BOUND,
    UPPER_BOUND,
    ESTIMATE,
    INITIALIZATION_PRIOR_TYPE,
    INITIALIZATION_PRIOR_PARAMETERS,
    NORMAL,
    LIN,
    TIME,
)


from .sbml_handler import SbmlHandler
from .petab_handler import PetabHandler
from .neural_network import NeuralNetwork, Layer, Node

class Model:
    def __init__(self, petab_problem: petab.Problem):
        self.petab_problem0 = petab_problem

        self.system_parts = []
        self.reset()

        self.mechanistic_rhs = self.sbml_handler.mechanistic_rhs
        self.universal_rhs = self.sbml_handler.universal_rhs

    def reset(self):
        petab_problem = self.get_petab_problem()
        self.sbml_handler = SbmlHandler(
            libsbml_model=petab_problem.sbml_model,
        )
        self.petab_handler = PetabHandler(
            petab_problem=petab_problem,
        )

    def get_petab_problem(self):
        return copy.deepcopy(self.petab_problem0)

    def add_neural_network(
        self,
        neural_network: NeuralNetwork,
        non_negative: bool,
        non_negative_bounded: bool,
        non_negative_prefactor: float = 1,
    ):
        """
        Updates the SBML model with a neural network.

        :param neural_network:
        :param non_negative: If true, uses the non-negative ude formulation x * NN
        :param non_negative_bounded: If true uses a the non-negative ude formulation with tanh bounds tanh(x) * NN
        :param non_negative_prefactor: The factor by which to multiply the state variable when non_negative_bounded = True: 
            tanh(prefactor * x) * NN
        """
        for layer in neural_network.layers:
            self.compute_layer_node_formulae(layer=layer)

        output_layer = neural_network.layers[-1]
        for output_node, output_species_id in zip(output_layer.nodes, neural_network.output_species_ids):
            kinetic_formula = f"{output_node.formula}"
            if non_negative:
                kinetic_formula = f"{output_species_id} * ({kinetic_formula})"
            elif non_negative_bounded:
                kinetic_formula = f"tanh({non_negative_prefactor} * {output_species_id}) * ({kinetic_formula})"

            neural_network_parameter_id = f"{output_species_id}__U__{output_node.id_}"

            self.sbml_handler.create_parameter_with_assignment(
                parameter_id=neural_network_parameter_id,
                formula=kinetic_formula,
            )

            self.sbml_handler.add_universal_term(
                species_id=output_species_id,
                parameter_id=neural_network_parameter_id,
            )
            
    def compute_layer_node_formulae(
        self,
        layer: Layer,
    ):
        for node in layer.nodes:
            bias_id = node.id_ + f"__bias__"
            self.create_affine_parameter(parameter_id=bias_id)
            node_formula_pieces = [f"({bias_id})"]
            for input_node in node.input_nodes:
                weight_id = node.id_ + f"__weight__" + input_node.id_
                self.create_affine_parameter(parameter_id=weight_id)

                input_node_formula = input_node.formula
                if input_node.input_nodes is not None:
                    input_node_formula = input_node.id_ + "__formula__"

                node_formula_piece = f"({weight_id} * ({input_node_formula}))"
                node_formula_pieces.append(node_formula_piece)
            node_formula = " + ".join(node_formula_pieces)
            node_formula = node.activation_function(node_formula)
            node.formula = node_formula

            self.sbml_handler.create_parameter_with_assignment(
                parameter_id=node.id_ + "__formula__",
                formula=node_formula,
            )

    def create_affine_parameter(
        self,
        parameter_id: str,
    ):
        self.sbml_handler.create_parameter(
            parameter_id=parameter_id,
            parameter_value=0.0,
            constant=True,
        )
        self.petab_handler.add_parameter(
            data={
                PARAMETER_ID: parameter_id,
                NOMINAL_VALUE: 0.0,
                PARAMETER_SCALE: LIN,
                LOWER_BOUND: -10.0,
                UPPER_BOUND: 10.0,
                ESTIMATE: 1,
                INITIALIZATION_PRIOR_TYPE: NORMAL,
                INITIALIZATION_PRIOR_PARAMETERS: "0;0.0001",
            },
        )
