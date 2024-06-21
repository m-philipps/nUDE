from typing import List, Union
import libsbml

from amici.sbml_import import _check_lib_sbml_errors

DIMENSIONLESS = "dimensionless"


class SbmlHandler:
    def __init__(
        self,
        libsbml_model: libsbml.Model,
    ):
        self.libsbml_model = libsbml_model

        self.libsbml_document = self.libsbml_model.getSBMLDocument()

        self.convert_reactions_to_rate_rules()
        self.mechanistic_rhs, self.universal_rhs = self.convert_rate_rules_to_parameters()

    def convert_reactions_to_rate_rules(self):
        if not len(self.libsbml_model.getListOfReactions()):
            return

        print("Converting reactions to rate rules...")

        # see libsbml CompFlatteningConverter for options
        conversion_properties = libsbml.ConversionProperties()
        conversion_properties.addOption("replaceReactions")
        if self.libsbml_document.convert(conversion_properties) != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise ValueError(
                "Error while converting reactions to rate rules."
            )

        self.libsbml_document.validateSBML()
        _check_lib_sbml_errors(
            sbml_doc=self.libsbml_document,
            show_warnings=True,
        )

    def convert_rate_rules_to_parameters(self):
        mechanistic_rhs = []
        universal_rhs = []

        for state_variable_id in self.state_variable_ids:
            mechanistic_formula = "0"

            rate_rule = self.libsbml_model.getRateRule(state_variable_id)
            if rate_rule is not None:
                mechanistic_formula = rate_rule.getFormula()
            else:
                rate_rule = self.libsbml_model.createRateRule()
                rate_rule.setVariable(state_variable_id)

            mechanistic_parameter_id = f"rhs_{state_variable_id}__mechanistic"
            universal_parameter_id = f"rhs_{state_variable_id}__universal"

            for parameter_id, formula in [
                [mechanistic_parameter_id, mechanistic_formula],
                [universal_parameter_id, "0"],
            ]:
                self.create_parameter(
                    parameter_id=parameter_id,
                    parameter_value=0.0,
                    constant=False,
                )
                self.create_assignment_rule(
                    variable=parameter_id,
                    formula=formula,
                )

            rhs = mechanistic_parameter_id + " + " + universal_parameter_id
            math = libsbml.parseL3Formula(rhs)
            rate_rule.setMath(math)

            mechanistic_rhs.append(mechanistic_parameter_id)
            universal_rhs.append(universal_parameter_id)

        return mechanistic_rhs, universal_rhs

    def create_parameter_with_assignment(
        self,
        parameter_id: str,
        formula: str,
    ):
        self.create_parameter(
            parameter_id=parameter_id,
            parameter_value=0.0,
            constant=False,
        )
        self.create_assignment_rule(
            variable=parameter_id,
            formula=formula,
        )

    def create_parameter_with_rate(
        self,
        parameter_id: str,
        formula: str,
        initial_condition: str = None,
    ):
        self.create_parameter(
            parameter_id=parameter_id,
            parameter_value=0.0,
            constant=False,
        )
        if initial_condition is not None:
            self.create_initial_assignment_rule(
                variable=parameter_id,
                formula=initial_condition,
            )
        self.create_rate_rule(
            variable=parameter_id,
            formula=formula,
        )

    def create_parameter(
        self,
        parameter_id: str,
        parameter_value: float,
        constant: bool,
    ):
        parameter = self.libsbml_model.createParameter()
        parameter.setId(parameter_id)
        parameter.setConstant(constant)
        parameter.setUnits(DIMENSIONLESS)
        parameter.setValue(parameter_value)
        return parameter

    def create_assignment_rule(
        self,
        variable: str,
        formula: str,
    ):
        assignment_rule = self.libsbml_model.createAssignmentRule()
        assignment_rule.setVariable(variable)
        math = libsbml.parseL3Formula(formula)
        assignment_rule.setMath(math)
        return assignment_rule

    def create_initial_assignment_rule(
        self,
        variable: str,
        formula: str,
    ):
        rule = self.libsbml_model.createInitialAssignmentRule()
        rule.setVariable(variable)
        math = libsbml.parseL3Formula(formula)
        rule.setMath(math)
        return rule

    def create_rate_rule(
        self,
        variable: str,
        formula: str,
    ):
        rule = self.libsbml_model.createRateRule()
        rule.setVariable(variable)
        math = libsbml.parseL3Formula(formula)
        rule.setMath(math)
        return rule

    @property
    def state_variable_ids(self):
        if not getattr(self, "_state_variable_ids", False):
            self._state_variable_ids = [s.getId() for s in self.libsbml_model.getListOfSpecies()]
        return self._state_variable_ids

    def add_universal_term(self, species_id, parameter_id):
        rhs_index = self.state_variable_ids.index(species_id)
        universal_parameter_id = self.universal_rhs[rhs_index]
        assignment_rule = self.libsbml_model.getAssignmentRule(universal_parameter_id)
        if assignment_rule is None:
            raise ValueError

        formula0 = assignment_rule.getFormula()
        formula = formula0 + " + " + parameter_id

        math = libsbml.parseL3Formula(formula)
        assignment_rule.setMath(math)

    def create_estimated_initial_concentration(
        self,
        species_id: str,
        parameter_id: str,
    ):
        parameter = self.create_parameter(
            parameter_id=parameter_id,
            parameter_value=0.0,
            constant=True,
        )
        
        initial_assignment = self.create_initial_assignment(
            symbol=species_id,
            formula=parameter_id,
        )

        return parameter_id, initial_assignment_id


    def create_initial_assignment(
        self,
        symbol: str,
        formula: str,
    ):
        initial_assignment = self.libsbml_model.createInitialAssignment()
        initial_assignment.setSymbol(symbol)

        math = libsbml.parseL3Formula(formula)
        initial_assignment.setMath(math)

        return initial_assignment
