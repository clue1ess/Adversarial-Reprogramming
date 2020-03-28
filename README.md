# Adversarial-Reprogramming

Implementation of paper : https://arxiv.org/pdf/1806.11146.pdf

### How to run

First, to train CIFAR10 dataset :<br>
List of arguments required :<br>
- batch_size of CIFAR10 images
- number of epochs
- learning rate
- momentum
- step_size for adaptive learning rate 
- gamma <br><br>
Run the following:<br><br>
`python3 cifar.py [batch_size] [num_epochs] [lr] [momentum] [step_size] [gamma]`<br><br>
Or run the following for default parameters<br><br>
`python cifar.py `

Now for reprogramming: <br>
List of arguments required :<br>
- batch_size of CIFAR10 images
- number of epochs
- learning rate
- momentum
- adversarial image size
- filename for loading CIFAR trained model (filename which will be output of first command)<br><br>
Run the following:<br><br>
`python3 reprogram.py [batch_size] [num_epochs] [lr] [momentum] [img_size] [filename]`<br><br>
Or run the following for default parameters<br><br>
`python reprogram.py [filename]`
