# Published example: Boehm with universal pApA export, NUDE Scenario 1

- FNN is a part of the ODE rhs

## Model structure

Boehm model for STAT5 dimerisation with a FNN for one of the interaction terms. The FNN, highlighted in red, replaces the `pApA_export` parameter, only the two species directly modified by the FNN are shown:

$$
\begin{align*}
    % STAT5A
    \frac{d\ \text{STAT5A}}{dt} =& - \text{BaF3\_Epo} \cdot \text{[STAT5A]}^2 \cdot \text{k\_phos} \\ 
    &-\ (\text{BaF3\_Epo} \cdot \text{[STAT5A]} \cdot \text{[STAT5B]} \cdot \text{k\_phos}) \\
    &+\ \textcolor{red}{ N[1] } \\
    &+ \text{k\_exp\_hetero} \cdot \text{[nucpApB]} \\
    % nucpApA
    \frac{d\ \text{nucpApA}}{dt} =& \text{k\_imp\_homo} \cdot \text{[pApA]} +\ \textcolor{red}{ N[2] } \\
\end{align*}
$$

To Do: update SBML
- [ ] add a parameter to be replaced: `pApA_export_in`, `pApA_export_out`
- [ ] convert reaction-based SBML to differential?

## Data-Driven Model Structure

The input to the FNN is the `nucpApA` species, outputs are the `pApA_export_in` and `pApA_export_out` placeholder parameters. Specification:

| LayerID | Layer                                                  | Activation Function |
|---------|--------------------------------------------------------|---------------------|
| layer1  | Linear(in_features = 1, out_features = 5, bias = true) | tanh                |
| layer2  | Linear(in_features = 5, out_features = 5, bias = true) | tanh                |
| layer3  | Linear(in_features = 5, out_features = 2, bias = true) | tanh                |

To Do: Was there an identity in the last layer?
