# Data-Driven Decoupling of Multivariate Functions

This repository contains the MATLAB implementation of my Bachelor's thesis at Maastricht University, titled **"Data-Driven Decoupling of Multivariate Functions"**.

## Overview

The project explores how to decouple multivariate functions into univariate components using only input–output data. It extends the symbolic method of Dreesen et al. (SIMAX 2015) to a fully data-driven setting by estimating Jacobian tensors and applying Canonical Polyadic Decomposition (CPD).

## Features

- Symbolic and numerical Jacobian estimation
- Local regression (linear, ridge, polynomial)
- Global polynomial regression 
- Tensor construction and CP decomposition using Tensorlab
- Experiments on functions of increasing complexity

## Requirements

- MATLAB R2021a or later
- [Tensorlab 3.0](https://www.tensorlab.net/)

