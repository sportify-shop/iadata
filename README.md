Welcome to a different type and part of the project. 
You will find there the first step of a group of beginners in the data scientist field. 

We have worked with : 

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) ![Scipy](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)

Quick start : 

Be sure you have already install Python on your computer.

```
git clone git@github.com:sportify-shop/iadata.git
```

Enter into the project directory and then create your own env by running (you can replace "myenv" with any name you prefer): 

```
python -m venv myenv
``` 

Now you will have to activate this environment with the following command: 

For Windows:
```
myenv\Scripts\activate
```

For macOS and Linux:
```
source myenv/bin/activate
```

Finally you will have to install some libraries to manipulate, visualise and analyse datas : pandas, numpy, plotly, scipy and IPython to get an outcome from running the index.py script. 
Hence run (make sure the Python's package manager `pip` is already installed): 

```
pip install pandas
pip install numpy
pip install plotly
pip install scipy
pip install ipython
```

We have retrieved a kaggle's dataset about house prices (the dataset is located in the house-price-advanced-regression-techniques folder more specifically within the train.csv file). 

There you get different type of figures depending on the data you want to manipualte, visualise and retrieve as a result by running : 

```
python index.py
// depending on your python's version installed
python3 index.py
```

# Done:
You should get a summary_test.html document and open less than a dozen of figures where different parameters and values are compared.

# Next Features:
Our work ends there. Next step will be to create a pipeline, build a model and test it, make some adjustements if needed, until he will seems the best to you. 

