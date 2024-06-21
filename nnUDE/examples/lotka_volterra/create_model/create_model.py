from yaml2sbml import YamlModel

yaml_model = YamlModel()


yaml_model.add_parameter(
    parameter_id="alpha_",
    nominal_value="1.3",
)
yaml_model.add_parameter(
    parameter_id="beta_",
    nominal_value="0.9",
)
yaml_model.add_parameter(
    parameter_id="gamma_",
    nominal_value="0.8",
)
yaml_model.add_parameter(
    parameter_id="delta_",
    nominal_value="1.8",
)

yaml_model.add_assignment(
    assignment_id="prey_birth",
    formula="alpha_ * prey",
)
yaml_model.add_assignment(
    assignment_id="prey_predation",
    formula="-beta_ * prey * predator",
)
yaml_model.add_assignment(
    assignment_id="predator_predation",
    formula="delta_ * prey * predator",
)
yaml_model.add_assignment(
    assignment_id="predator_death",
    formula="-gamma_ * predator",
)

yaml_model.add_ode(
    state_id="prey",
    right_hand_side="prey_birth + prey_predation",
    initial_value=0.44249296,
)
yaml_model.add_ode(
    state_id="predator",
    right_hand_side="predator_predation + predator_death",
    initial_value=4.6280594,
)

yaml_model.write_to_sbml("model.xml")
