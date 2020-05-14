# Multivariate time series classification and application for real time navigation
This is the code repository of my dissertation project concerning multivariate time series classification. 
The project was undertaken as partial fulfillment of my BSc Mathematics & Computer Science degree at the University of Lincoln.
# What is this?
The project is concerning whether the traversability of a level crossing can be approximated by a classifier based on observed data. Modern navigation applications do not know the difference between
a level crossing that is traversable and one that is not; in many cases, drivers could be rerouted away from level crossings to save them time, especially when the classifier recognises that an unusually long waiting time
is to be anticipated. The data the classifier based on is a simulated time series of how a level crossing's barriers react relative to train traffic. This was realised by gathering arrival times of trains, which were later transformed
to the time series using an algorithm. The time series data is most accurately captured by some sort of IoT device (like a sensor near tracks or a camera using computer vision) but this was not possible in this case. 
Following this I have implemented various classifiers based on the generated data which demonstrate that the traversability of level crossings can be modelled and this repository contains the code I have used to develop the classifiers as well as some 
scripts used to generate some graphs for the written document. 
# Installation and running
There are many dependencies, the `install-all.bat` file runs the following commands, installing all dependencies (or you may install manually):
```
pip install numpy
pip install matplotlib
pip install peewee
pip install schedule
pip install pandas
pip install seaborn
pip install sklearn
pip install tensorflow
pip install gpflow
pip install keras
pip install progress
```
Following this all code should be executable using `Python 3.8.0`. It is suggested that the scripts are ran using the command line (by double clicking on the script) especially for the classifier scripts (`analyse-*.py`) because of progress bar support and better performance. The `.db` file is an [sqlite database](https://www.sqlite.org/index.html) which I interacted with using [DBeaver](https://dbeaver.io/).

# Structure
To explain briefly, the method was as follows:
1. Gather data of trains on a train stop which is right next to a level crossing (I have used Lincoln Central)
2. Convert to time series format using an algorithm
3. Load the time series to a dataframe, preprocess using feature engineering
4. Create classifier



Each of these tasks are achieved by separate script files, here is what they each do:
- `init_db.py` initialises a database, which can be used later on. This is where the train information is saved, `trains.db`.
- `scraper.py` is the file responsible for scraping live information from the NationalRail website. To change the station(s) monitored, change the array of URLs in the body. 
- `Simulation.py` processes the `trains.db` file into a time series placed in `\datastream\` folder, broken up by days of the week. 
- `create_load_df.py` is a snippet which loads the `\datastream\` data, processes it and saves it **or** loads a `pandas` dataframe saved in a `.h5` file. This file is imported by relevant scripts rather than being used on its own.
- `backup_db.py` is the script responsible for sending a copy of a file to an email address.
- `analyse-binary-*` are used to create and evaluate the classifiers. If you choose to load a model, use **models/bcm** when prompted for the name of the file. 
- `analyse-multivariate-*` are used to create and evaluate the multi class classifiers. In this case use **models/mcm** when prompted for the name of the file, unless you opt to generate a new model. 