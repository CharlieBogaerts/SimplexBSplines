# Simplex B Splines

## General info
This project contains scripts that allow for estimating a simplex B-splines model from data. The theory is taken from the lecture slides of the course AE4320 System Identification of Aerospace Vehicles as part of the Master Control and Simulation on the faculty of Aerospace Engineering of Delft University of Technology. Theory can be found in papers from Coen C. de Visser, eg:
- C.C. de Visser, Q.P. Chu, J.A. Mulder 'A new approach to linear regression with multivariate splines' (Automatica, 2009)
- C.C. de Visser, Q.P. Chu, J.A. Mulder 'Differential constraints for bounded recursive identification with multivariate splines' (Automatica, 2011)

The model estimates a dependent variable Y as function of an N dimensional independent variable X. Note that for higher dimensions finding a proper mesh can be challenging, since the model can not deal with degenerate simplices (simplices with all cornerpoints in the same hyperplane).

## How to use
The folder 'examples' contains scripts that use almost all the functions that are available. These consist of estimating a model from data, saving the model, opening a saved model and evaluating the simplex spline at a given set of vertices.
