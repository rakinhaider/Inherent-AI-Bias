# Unfair AI: It Isn't Just Biased Data

## Description

In this project, we demonstrate AI systems specifically Bayesian 
Classifier has the capacity to induce unfairness in outcome which
is otherwise absent in the data. We train Gaussian Naive Bayes
classifiers on synthetic fair balanced dataset *(SFBD)* of 10000
samples with 2 or 10 features. Results show that Naive Bayes models
can produce unfair outcome even with fair balanced dataset. We 
also experiment with real-world dataset, namely, COMPAS. After
de-biasing with method proposed in [[1]](#1), we resample to obtain 
relatively fair balanced dataset.

## Getting Started

### Pre-requisites

The codes are built and tested with the following 
system configurations.

|  Item   |  Version                      |
| ------- | ------------------------------|
| OS      | Linux (Ubuntu 20.04.2 64bit)  |
| Python  | 3.8.10                        |
| pip     |  20.0.2                             |

### Randomization Seeds
The following seeds where used to ensure reproducibility of the results. Further experiments can be carried out by trying out different randomization seeds.

|  Case                         |  Seed    |
| ------- | ------------------------------|
| Training SFBD Generation      |   47  |
| Test SFBD Generation          |   41  |
| COMPAS train test split       |   23  |

### Getting Started

Create a virtual environment and activate it with the following
commands.

```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

### Installing Dependencies

Install all the dependencies with the following command.

```bash
pip3 install -r requirements.txt
```

After installation, you need to download the COMPAS dataset
and put it in **data/raw/compas/** inside the aif360 folder. We provide 
*dowload.py* to perform this task. Run the following program
by replacing *<aif360-folder>* with the aif360 folder location.

```bash
python3 download.py --aif360-folder <aif360-folder>
```
In my case it was,
```bash
python3 download.py --aif360-folder myvenv/lib/python3.8/site-packages/aif360/
```


### Executing program

* To generate all the results use *run.sh*
* The following command will generate all the tables in the paper
```bash
chmod +x run.sh
./run.sh
```

* You can also run the script individually to obtain the results.
For example, the following command will output model performances
for less separable unprivileged group with 2 attributes.
```bash
python3 -m with_resource_constraints
```
## Help

Please ignore **tensorflow** related warnings.

## Authors

Left blank for review purpose.

## Reference
<a id="1" href="https://arxiv.org/abs/2105.04534">[1]
Improving Fairness of AI Systems with Lossless De-biasing. arXiv preprint arXiv:2105.04534.</a> 