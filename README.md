# T-SAIRUS

Code for the T-SAIRUS method. Here is some information on how to use it.
### Install requirements
By the command 

``` pip install -r requirements.txt ```

The code has been written in **Python 3.8**

### Prepare data
T-SAIRUS performs user classification on an evolving network. The evolution of the network is represented via temporal
snapshots, meaning that there are multiple snapshots of the network. From a snapshot to the other there may be nodes
that are added, deleted or modified, as well as the relations. In the experiments that we run we used a network divided
in **5** then **10** snapshots.

We use the first N-1 snaps for training and the last one for testing. The learning is supervised. Since the learning 
takes into account the content posted by the users, their social relationships and their spatial relations, the data 
needs to be structured in this way:

```
dataset
    |__snap1
        |__ content_dataframe.csv
        |__ social_network.dat
        |__ spatial_network.dat
    |__snap2
        |__ content_dataframe.csv
        |__ social_network.dat
        |__ spatial_network.dat
    ...
    |__snapN
        |__ content_dataframe.csv
        |__ social_network.dat
        |__ spatial_network.dat
```
Each snap will contain information related to that snap, but it's important that the files containing content and the 
networks have the same name in the snaps. The ```dataset/example``` directory contains a simple example dataset with
three splits. Its only purpose is to show an example of how the dataset should be structured

### Content file structure
The csv containing the content must have three columns:
1. One containing the user ID
2. One containing the text corresponding to the user. Keep in mind that the text must be like a single, big string, and
not a list of words.
3. One containing the label
You can choose the column names, as long as you specify them in the ```parameters.yaml``` file.

The text column contains, for each user, the concatenation of his posts, falling in the timestamps that delimit the
current snapshot. The preprocessing operations I performed are lemmatization, stemming, stopwords removal. The file
```modelling/text_preprocessing``` contains the code for preprocessing the text.

### Network file structure
The networks must be saved in tab separated files. The files can have any name and any extension, you simply need to 
specify the name in the yaml.  

### Social network
The social network (ie the network describing the social relations such as *follow*) depict one relationship per row. 
For instance, if we have two users with ID ```150``` and ```151```, and the former follows the latter, then the file
will have the following row:
```150\t\t151```
If the relation is asymmetric, then another row will contain the opposite relation:
```151\t\t150```

### Spatial network
The spatial network depicts the spatial relationships among users. It has a structure similar to the one of the social
network, but each row contains three values: the user IDs and their closeness. We represent the closeness as the
geodetic distance among the users' positions, computed using their coordinates. Suppose to have the following row:

```151\t\t150\t\t0.89```

It means that the two users have a closeness value equal to ```0.89``` 