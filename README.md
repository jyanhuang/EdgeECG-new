# EdgeECG

This repository provides the implementation of the proposed EdgeECG for arrhythmia classification, including Python code, C implementation, and STM32F103ZET6 deployment.


## Repository Structure

- `EdgeECG/Python/`  
  Training and evaluation code for the proposed model.

- `EdgeECG/C_Implementation/`  
  Self-contained C implementation of the neural network inference, including:
  - model definition  
  - weight parameters  
  - example inference entry  

- `EdgeECG/STM32_Project/`  
  Example project for deployment on STM32F103ZET6 microcontrollers.
