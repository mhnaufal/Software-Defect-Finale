# Software Defect Prediction

Machine learning model for 'Software Defect Prediction' with a combination using deep learning and based on Python & Tensorflow

## General Informations

- Datasets
  
  Datasets for this project taken from [PROMISE public dataset](http://promise.site.uottawa.ca/SERepository/datasets-page.html)

- Models
  
  CNN, RNN, LSTM, Random Forest, and more

- Results
  
  _Soon_
  
  _Results from running model(s) shown in results folder_

---

## Project Structure

```
│datasets
├── processed
    └── ...
├── raw
    └── ...

│references
└── README.md

│reports
├── figures
    └── ...
├── results
    └── ...
└── preprocess.txt

│src
├── models
│   ├── cnn.py
│   ├── lstm.py
│   ├── landom_forest.py
│   └── ...
├── main.py
└── preprocess.py

│README.md
│requirements.txt
```

---

## Installation

_Windows 10 steps_

Clone this repository or download it manual as zip

```bash
$ git clone https://github.com/mhnaufal/Software-Defect-Finale.git
```

Open up **cmd** or **Powershell** (Powershell prefered) as Administrator and move to inside this repo directory

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

## Credits

Inspired by a lot of other research and study listed in [here](https://github.com/mhnaufal/Software-Defect-Finale/tree/main/references)

## License

[MIT](LICENSE)
