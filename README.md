Robust blood Detection Model through weakly supervised localization, self-supervised pretraining, adversarial training, 3D convolutions, and video frame modeling.
Overview
This repository provides a PyTorch deep neural network for classifying microscope images of blood cells as benign or malignant. The model achieves high accuracy and generalizability by leveraging:
Weakly supervised localization to identify explanatory regions used by the classifier via class activation mappings
Self-supervised pretraining on unlabeled blood cell video data to prime feature extraction layers
Test-time adversarial training to improve model robustnes to small input perturbations
3D convolutions to analyze volumetric shape cues rather than flattened 2D images
Video frame order modeling for additional temporal self-supervision
Combined, these techniques improve predictive performance while providing intepretability.

-Installation
This code requires Python 3.8+ and Poetry for dependency management. Install dependencies with:

poetry install

-Activate the Poetry environment for usage:
poetry shell

-Usage
To train the model on the blood_cell_videos dataset:
python training/trainer.py --data-dir /path/to/blood_cells --epochs 100 --lr 0.001

This will configure the neural architecture, leverage unlabeled videos for self-supervision, and fit the model parameters on the labeled dataset.

-Contributing
Contributions to improving the weakly supervised and self-supervised components are greatly welcome! Please open issues for any bugs or desired functionality.

-License
This project is licensed under the MIT license. See LICENSE.md for details.
Let me know if you would like any additional sections or more information added!