# ☠ Software Defect Prediction

Machine learning model for _'Software Defect Prediction'_ using deep learning and based on Python & Tensorflow

## Introduction

- Datasets
  
  Datasets for this project taken from [PROMISE public dataset](http://promise.site.uottawa.ca/SERepository/datasets-page.html)

- Models
  
  CNN, RNN, LSTM, Random Forest, and more

- Results
  
  _-Soon-_
  
  _Results from running model(s) shown in reports folder_

---

## Project Structure

```
│datasets
├── processed
│   └── big_data1.csv
│   └── big_data2.csv
│   └── ...
├── raw
    └── ...

│references
└── README.md

│reports
├── figures
│   └── confussion matrix
│       └── random_forest.png
│       └── cnn.png
│   └── ...
├── results
│   └── random_forest.txt
│   └── ...
└── preprocess.txt

│src
├── models
│   ├── cnn.py
│   ├── lstm.py
│   ├── random_forest.py
│   └── ...
├── main.py
└── preprocess.py

│README.md
│requirements.txt
```

---

## Installation

💻 _Windows 10 steps_

Clone this repository or download it manual as a zip

```bash
$ git clone https://github.com/mhnaufal/Software-Defect-Finale.git
```

Open up **cmd** or **Powershell** (Powershell prefered) as Administrator and go to this repo directory

Create Python virtual environment:

```python
$ python -m venv sddl-env
```

Run the virtual environment:

```python
$ sddl-env/Scripts/activate
```

Install the library:

```python
$ pip install -r requirements.txt
```

_If above command result an error, run the cmd or Powershell as Administrator and then re run the above command_

Run the models:

```python
$ python src/main.py
```

or

```python
$ python src/models/random_forest.py
```

## Credits

Inspired by many other studies listed in [here](https://github.com/mhnaufal/Software-Defect-Finale/tree/main/references)

## License

[MIT](LICENSE)
