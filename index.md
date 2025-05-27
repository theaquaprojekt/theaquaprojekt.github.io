## Automating Experimental Optics with Sample Efficient Machine Learning Methods

**Paper Link:** [arXiv:2503.14260](https://arxiv.org/abs/2503.14260)  
**For more information:** [GitHub Repository](https://github.com/arindam5aha/aqua)

### Visualising the Observation Space

### Visualising Different Hermite-Gauss Modes
These outputs are simulated.

![png](Final_Plots_files/Final_Plots_5_0.png)

### Calculating Finesse from Full FSR Trace

Analysing both transmission and reflection traces:

- Top two peak positions: [1.41426484, 1.42037134]
- FWHM of the top two peaks: [4.375, 3.50930248]
- Difference between top two peak positions: 504
- Ratio of difference to FWHM: [115.2, 143.618284]

Finesses used for paper: 129.40914200074948

### Control Specs

Bounds for the optimization and control:  
The values represent the thread counts moved by the actuators.  
For more info visit: [Thorlabs](https://www.thorlabs.com/navigation.cfm?guide_id=83)

- Lens_1: (-100000, 100000)
- Lens_2: (-100000, 100000)
- Mirror_1x: (-5000, 5000)
- Mirror_1y: (-5000, 5000)
- Mirror_2x: (-5000, 5000)
- Mirror_2y: (-5000, 5000)

### Reward Function Used in Experiment: $\eta$

This is the sole reward function used in the experiment for both SANN and AQUA.

```python
def noisy_eta(obs, target_region=(400, 550)):
    #input: obs - the observed cavity trace (array of length 1024)
    #target_region - the region of interest, including the target mode
    
    target_integral = np.trapz(obs[target_region[0]:target_region[1]])
    complete_integral = np.trapz(obs)
    eta = target_integral / complete_integral
    return eta
```

### Corrected Reward Function: $\eta'$

Used to post-process all collected data and use for comparison.

```python
def corrected_eta(obs, distance=5, prominence=5e-3, target_region=(400, 550)):
    # input: distance, prominence are parameters for the SciPy peak finding algorithm
    # these are to be kept constant for all data, unless additional corrections are needed.
    
    # Correct for the detector offset
    obs = obs - np.median(obs)
    
    # Find the peaks    
    peaks, _ = find_peaks(obs, distance=distance, prominence=prominence)    

    # Calculating the reward
    target_pk_height = max(obs[target_region[0]: target_region[1]])
    sum_pk_height = np.sum(obs[peaks])
    eta_prime = target_pk_height / sum_pk_height
    return eta_prime
```

## Results and Discussion

### Manual Alignment

![png](Final_Plots_files/Final_Plots_18_0.png)

![png](Final_Plots_files/Final_Plots_19_0.png)

- MAX HUMAN MME: 0.9565508021390374
- MEAN HUMAN MME: 0.9319114506522006
- Std. Dev: 1.6788265680236416 %

<!-- ### SANN Optimisation on Experiment

![png](Final_Plots_files/Final_Plots_23_0.png)

Visualising best SANN results.

![png](Final_Plots_files/Final_Plots_25_0.png) -->

### Analyse AQUA Performance on Experiment

Loading run with resets.

Loading No Reset Run.

![png](Final_Plots_files/Final_Plots_31_0.png)

- Avg time per step: 5.435340497249661

![png](Final_Plots_files/Final_Plots_33_0.png)

- MAX HUMAN MME: 95.66 %
- MEAN HUMAN MME: 93.19 %

### Supplementary Information

### Linear Relation Between $\eta$ and $\eta'$

Each colored scatter plot represents an individual SANN run. The runs occurred on different days and as can be seen from the plot below, each run gives a slightly different slope suggesting change in conditions. Deviations seen are caused by noise. The detector offset causes the x-axis to have an offset, meaning $\eta$ does not reach 0.

- Slope: 1.78561321556722
- Intercept: -0.11827958713759489

![png](Final_Plots_files/Final_Plots_36_1.png)

### Thermal Drift

Fluctuations in target peak heights and its position.

- Best Cost Height: 1.7208110246574506
- Best Cost: 0.6379340335853348
- Transmittance: 1.7256962227402255

![png](Final_Plots_files/Final_Plots_38_1.png)

### Actuator Drifts

![png](Final_Plots_files/Final_Plots_40_0.png)

### Scatter Plots of the Parameter Space

![png](Final_Plots_files/Final_Plots_43_0.png)

### AQUA: Model Specs

- Observation size: 1024, raw data as shown above without offset corrections is used (continuous)
- Action size: 6, uses the full bounds as mentioned previously (continuous)

Models config:
- Compatible with PyTorch version: 2.1.2
- CUDA version: 12.1
- [PyTorch](https://pytorch.org/)

- hidden_size: {'encoder': 64, 'prediction': 64, 'policy': 512}
- num_hidden_layers: 2
- latent_size: 32
- activation: leakyRelu
- input_sizes: {'encoder': '1024 + (6)', 'prediction': '1024 + (6)', 'policy': '32 + (6)'} // (6) for scaled parameters optional or used with dropouts
- optimiser: Adam
- lr: 0.0001
- batch_size: {'encoder': 50, 'prediction': 50, 'policy': 200}
- clip_grad_norm: 1.0
- dropout: 0.2
- weights_initializer: kaiming_uniform

Total no. of trainable parameters in the models: 558416  
(The latest version has 10x less parameters and much improved generalisation and training times)

### AQUA: Pre-training

### AQUA: Reset Conditions

All resets shown in the paper apply a parameter chosen randomly from the entire given bounds.  
As seen in AQUA's online learning plot, these mostly return $\eta'=0$. Below we visualise the associated observations and the parameters.

![png](Final_Plots_files/Final_Plots_49_0.png)

![png](Final_Plots_files/Final_Plots_49_1.png)

### AQUA: Realign Without Retraining

![png](Final_Plots_files/Final_Plots_51_0.png)

### AQUA: Visualising Top 5% Rewarded Observations

![png](Final_Plots_files/Final_Plots_53_0.png)

### Cross Entropy Method (CEM) Sampling on Experiment

CEM is a pseudo-random importance sampling method.  
[DOI: 10.1023/A:1010091220143](https://doi.org/10.1023/A:1010091220143)

![png](Final_Plots_files/Final_Plots_55_0.png)

![png](Final_Plots_files/Final_Plots_56_0.png)

As can be seen here, the reward landscape is different, given different configurations of the optics.

![png](Final_Plots_files/Final_Plots_57_1.png)

### Compare AQUA with Standard Model-Free RL (Stable Baselines) in Cavity Simulation

It is to be noted that the simulations are free of all kinds of drifts and one can expect to get similar rewards for repeat measurements.  
Thanks to Viktoria-S. Schmiesing for help with setting up SB3 and its reward function and Aaron Tranter for the simulation.  
For more information on the simulation environment visit: [GW Optics](https://www.gwoptics.org/pykat/)  
[Stable Baselines](https://stable-baselines.readthedocs.io/en/master/)  
[OpenAI Gym](https://www.gymlibrary.dev/)

![png](Final_Plots_files/Final_Plots_62_0.png)

Reward Functions used in simulation, with additional penalties to account for the multi-step architecture of standard model-free RL.