# Measurement Simplification
This repo implements the code for [Measurement Simplification in ρ-POMDP with Performance Guarantees](https://arxiv.org/abs/2309.10701)

## Installation
Clone the repository:
```bash
git clone https://github.com/<username>/<repository>.git
```
Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
```python
python m_simplification.py
```

## Configuration
The behavior of the belief space planning system can be configured via the `config.py` file. Parameters such as the scenario, number of landmarks, map size, prior mapping, goal location, and number of paths can be adjusted according to specific requirements.

## Class: `MeasurementSimplification`
### Attributes:
- `prior`: The prior belief state.
- `actions_random`: A list of random actions.
- `actions`: A list of predefined actions.
- `scenario`: The scenario for generating landmarks.
- `num_landmarks`: The number of landmarks to generate.
- `map_size`: The size of the map.
- `landmarks`: A dictionary containing information about the landmarks.
- `belief`: The GaussianBelief object representing the robot's belief.
- `fig_num`: The figure number for plotting.


## Citation
If you use, compare with, or refer to this work, please cite:
```
@misc{yotam2023measurement,
      title={Measurement Simplification in \rho-POMDP with Performance Guarantees}, 
      author={Tom Yotam and Vadim Indelman},
      year={2023},
      eprint={2309.10701},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```