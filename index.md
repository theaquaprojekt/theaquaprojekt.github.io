## Automating Experimental Optics with Sample Efficient Machine Learning Methods

**Paper Link:** [arXiv:2503.14260](https://arxiv.org/abs/2503.14260)  
**For more information:** [GitHub Repository](https://github.com/arindam5aha/aqua)

### Abstract
As free-space optical systems grow in scale and complexity, troubleshooting becomes increasingly time-consuming and, in the case of remote installations, perhaps impractical. An example of a task that is often laborious is the alignment of a high-finesse optical resonator, which is highly sensitive to the mode of the input beam. In this work, we demonstrate how machine learning can be used to achieve autonomous mode-matching of a free-space optical resonator with minimal supervision. Our approach leverages sample-efficient algorithms to reduce data requirements while maintaining a simple architecture for easy deployment. The reinforcement learning scheme that we have developed shows that automation is feasible even in systems prone to drift in experimental parameters, as may well be the case in real-world applications.

<img src="Final_Plots_files/fig_2.svg" width="700"/>

### Visualising the Observation Space
<img src="Final_Plots_files/random_obs.gif" width="700"/>

### Visualising Different Hermite-Gauss Modes
These outputs are simulated.

<img src="Final_Plots_files/Final_Plots_5_0.png" width="700"/>

### Control Specs

Bounds for the optimization and control:  
The values represent the thread counts moved by the actuators.  
For more info visit: [Thorlabs](https://www.thorlabs.com/navigation.cfm?guide_id=83)

| Parameter   | Bounds                |
|-------------|----------------------|
| Lens_1      | (-100000, 100000)    |
| Lens_2      | (-100000, 100000)    |
| Mirror_1x   | (-5000, 5000)        |
| Mirror_1y   | (-5000, 5000)        |
| Mirror_2x   | (-5000, 5000)        |
| Mirror_2y   | (-5000, 5000)        |

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

<img src="Final_Plots_files/Final_Plots_18_0.png" width="700"/>
<img src="Final_Plots_files/Final_Plots_19_0.png" width="400"/>

- MAX HUMAN MME: 0.956
- MEAN HUMAN MME: 0.932
- Std. Dev: 1.68 %

<!-- ### SANN Optimisation on Experiment

<img src="Final_Plots_files/Final_Plots_23_0.png" width="700"/>

Visualising best SANN results.

<img src="Final_Plots_files/Final_Plots_25_0.png" width="700"/> -->

### Analyse AQUA Performance on Experiment
The AQUA algorithm is designed to address the challenges of automating optical alignment in experimental setups, where manual tuning is often inefficient and susceptible to environmental drifts. By leveraging a sample-efficient reinforcement learning approach, AQUA can adapt to changing system dynamics and maintain optimal performance with fewer experimental trials. The method is particularly suited for scenarios where data collection is costly or time-consuming, such as in high-finesse optical resonators. The results below illustrate AQUA's ability to achieve and sustain high alignment quality, outperforming traditional manual methods and demonstrating robustness against typical sources of experimental noise and drift.
<img src="Final_Plots_files/Final_Plots_31_0.png" width="700"/>

<img src="Final_Plots_files/Final_Plots_33_0.png" width="400"/>

- Avg time per step: 5.44 sec
- Human $\eta'$ (max) : 95.66 %
- Human $\eta'$ (mean) : 93.19 %

### Supplementary Information

### Linear Relation Between $\eta$ and $\eta'$

Each colored scatter plot represents an individual SANN run. The runs occurred on different days and as can be seen from the plot below, each run gives a slightly different slope suggesting change in conditions. Deviations seen are caused by noise. The detector offset causes the x-axis to have an offset, meaning $\eta$ does not reach 0.

- Slope: 1.79
- Intercept: -0.12

<img src="Final_Plots_files/Final_Plots_36_1.png" width="400"/>

### Thermal Drift

Fluctuations in target peak heights and its position.

<img src="Final_Plots_files/Final_Plots_38_1.png" width="700"/>

### Actuator Drifts

<img src="Final_Plots_files/Final_Plots_40_0.png" width="500"/>

### Scatter Plots of the Parameter Space

<img src="Final_Plots_files/Final_Plots_43_0.png" width="700"/>

### AQUA: Model Specs

- **Observation size:** 1024 (raw data, no offset correction; continuous)
- **Action size:** 6 (full bounds as previously described; continuous)
- **PyTorch version:** 2.1.2
- **CUDA version:** 12.1
- **Hidden sizes:** encoder: 64, prediction: 64, policy: 512
- **Number of hidden layers:** 2
- **Latent size:** 32
- **Activation:** leakyReLU
- **Input sizes:**  
  - encoder: 1024 + 6  
  - prediction: 1024 + 6  
  - policy: 32 + 6  
    (6 for scaled parameters, optionally used with dropouts)
- **Optimizer:** Adam
- **Learning rate:** 0.0001
- **Batch size:** encoder: 50, prediction: 50, policy: 200
- **Gradient clipping:** 1.0
- **Dropout:** 0.2
- **Weights initializer:** kaiming_uniform

**Total trainable parameters:** 558,416  
*(Latest version: 10x fewer parameters, improved generalization and training times)*

<!-- ### AQUA: Pre-training -->

### AQUA: Reset Conditions

All resets shown in the paper apply a parameter chosen randomly from the entire given bounds.  
As seen in AQUA's online learning plot, these mostly return $\eta'=0$. Below we visualise the associated observations and the parameters.

<img src="Final_Plots_files/Final_Plots_49_0.png" width="700"/>
<img src="Final_Plots_files/Final_Plots_49_1.png" width="700"/>

### AQUA: Realign Without Retraining

<img src="Final_Plots_files/Final_Plots_51_0.png" width="700"/>

### AQUA: Visualising Top 5% Rewarded Observations

<img src="Final_Plots_files/Final_Plots_53_0.png" width="700"/>

### Cross Entropy Method (CEM) Sampling on Experiment

CEM iteratively samples candidate solutions from a probability distribution, evaluates their performance, and updates the distribution to focus on the most promising regions of the parameter space. This method is particularly useful for optimization problems where the search space is large and gradients are unavailable or unreliable. In the context of optical alignment, CEM provides a baseline for comparison against more sophisticated reinforcement learning approaches like AQUA, highlighting the benefits of sample efficiency and adaptability in dynamic experimental environments.
[DOI: 10.1023/A:1010091220143](https://doi.org/10.1023/A:1010091220143)

<img src="Final_Plots_files/Final_Plots_55_0.png" width="700"/>
<img src="Final_Plots_files/Final_Plots_56_0.png" width="700"/>

As can be seen here, the reward landscape is different, given different configurations of the optics.

<img src="Final_Plots_files/Final_Plots_57_1.png" width="700"/>

### Compare AQUA with Standard Model-Free RL (Stable Baselines) in Cavity Simulation

It is to be noted that the simulations are free of all kinds of drifts and one can expect to get similar rewards for repeat measurements.  
Thanks to Viktoria-S. Schmiesing for help with setting up SB3 and its reward function and Aaron Tranter for the simulation.  
For more information on the simulation environment visit: [GW Optics](https://www.gwoptics.org/pykat/)  
[Stable Baselines](https://stable-baselines.readthedocs.io/en/master/)  
[OpenAI Gym](https://www.gymlibrary.dev/)

<img src="Final_Plots_files/Final_Plots_62_0.png" width="700"/>

Reward Functions used in simulation, with additional penalties to account for the multi-step architecture of standard model-free RL.

```python
# THE REWARD FUNCTION FOR AQUA THAT CALCULATES COST FROM A TRACE

def reward(self, obs):        
    pv, _ = find_peaks(self.best_obs.squeeze(), height=0)
    pm = 15
    p1 = np.trapz(obs[pv[0]-pm:pv[0]+pm].squeeze())
    not_p1 = np.trapz(np.concatenate([obs[:pv[0]-pm], obs[pv[0]+pm:]]).squeeze()) * 0.3
    obs_reward = ((p1 - not_p1) + 4.0)/10.0 
    
    # The constants are chosen such that the best reward is less than 1 (by choice, not a necessity)
    return obs_reward
    
# THE REWARD FUNCTION FOR SB THAT USES THE AQUA REWARD 
# WITH ADDITIONAL PENALTIES STEPS TAKEN AND BOUNDS CROSSED

def base_rwd(self, obs):
    # a static reward function   
    nc = reward(obs) 
    t = self.accuracy * self.max_neg_cost
    if nc >= t:
        rwd = 1.5*(np.exp(10*(nc-t))-1.0)
    else:
        rwd = (1.0-1.0/nc) + t
    return rwd

def calculate_reward(self, neg_cost):
    reward = 0
    bounds_crossed = False

    for i, p in enumerate(self.current_params):
        if p >= self.u_bounds[i] or p <= self.l_bounds[i]:
            bounds_crossed = True
    if bounds_crossed:
        reward -= 1

    if self.steps > self.max_steps:
        self.truncated = True

    reward += self.base_rwd(neg_cost)

    if neg_cost >= self.dynamic_accuracy * self.max_neg_cost:
        reward += 10*neg_cost
        self.success += 1
        self.dynamic_accuracy += 0.01
        if self.dynamic_accuracy >= 1.0:
            self.dynamic_accuracy = 1.0
        self.terminated = True

    return reward

    # NOTE: IN BOTH CASES (AQUA & SB) THE OBSERVATION GIVEN IS SAME (TRACE + PARAMS)
    #       AND THE BOUNDS USED ARE ALSO SAME
```