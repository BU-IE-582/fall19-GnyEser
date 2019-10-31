
# Homework 1 and 2

Importing necessary libraries.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
```

Reading the data from csv file:


```python
data = pd.read_csv("matches.csv")
data.fillna(0)
data.head()
home_goals  = list()
away_goals = list()
home_away_goals = list()
```

Let's grab the data for only English Premier League which its ID is 148:


```python
x = data['league_id'] == 148
data = data[x]
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_awayteam_id</th>
      <th>match_hometeam_id</th>
      <th>match_id</th>
      <th>epoch</th>
      <th>match_status</th>
      <th>match_live</th>
      <th>match_hometeam_name</th>
      <th>match_awayteam_name</th>
      <th>match_hometeam_score</th>
      <th>match_awayteam_score</th>
      <th>match_hometeam_halftime_score</th>
      <th>match_awayteam_halftime_score</th>
      <th>match_hometeam_extra_score</th>
      <th>match_awayteam_extra_score</th>
      <th>match_hometeam_penalty_score</th>
      <th>match_awayteam_penalty_score</th>
      <th>league_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2614</td>
      <td>2619</td>
      <td>13331</td>
      <td>1505561400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Crystal Palace</td>
      <td>Southampton</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2626</td>
      <td>2623</td>
      <td>13329</td>
      <td>1505570400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Watford</td>
      <td>Manchester City</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2629</td>
      <td>2621</td>
      <td>13327</td>
      <td>1505570400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Liverpool</td>
      <td>Burnley</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2641</td>
      <td>2654</td>
      <td>13456</td>
      <td>1505570400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Sheffield Utd</td>
      <td>Norwich</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2617</td>
      <td>2616</td>
      <td>13324</td>
      <td>1505651400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Chelsea</td>
      <td>Arsenal</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
    </tr>
  </tbody>
</table>
</div>



Assing home team goals and away team goals in seperate lists.


```python
home_goals = data["match_hometeam_score"].tolist()
away_goals = data["match_awayteam_score"].tolist()
```

Let's see how big our dataset.


```python
data["match_hometeam_score"].to_frame().info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 628 entries, 1 to 5161
    Data columns (total 1 columns):
    match_hometeam_score    618 non-null float64
    dtypes: float64(1)
    memory usage: 9.8 KB
    

Let's create another list for results combining home and away goals.


```python
for i in range(618):
    home_away_goals.append(str(home_goals[i]) + "-" + str(away_goals[i]))
```

Let's see the histogram of home team goals.


```python
sns.set(style="darkgrid")
plt.figure(figsize=(13,7))
ax = sns.countplot(x=home_goals)
ax.set(xlabel='Home Goals', ylabel='Number of Games')
plt.show()
```


![png](output_15_0.png)


And away goals:


```python
sns.set(style="darkgrid")
plt.figure(figsize=(13,7))
ax = sns.countplot(x=away_goals)
ax.set(xlabel='Away Goals', ylabel='Number of Games')
plt.show()
```


![png](output_17_0.png)


And which results are more likely to occur in matches:


```python
plt.figure(figsize=(17,7))
ax = sns.countplot(x=home_away_goals)
ax.set(xlabel='Home goals – Away Goals', ylabel='Number of Games')
plt.xticks(rotation=45)
plt.show()
```


![png](output_19_0.png)


It looks like distribution of away goals looks like poisson distribution.


```python
aways = data.match_awayteam_score.dropna().astype(int)
dats = list()
for i in aways:
    dats.append(int(i))
```


```python
from scipy import stats
from scipy.stats import norm
```

When we try to plot distribituon :


```python
sns.distplot(data.match_awayteam_score.dropna(), bins=7)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x234778f3160>




![png](output_24_1.png)


# Task 2


```python
books = pd.read_csv("booking.csv")
books.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>time</th>
      <th>home_fault</th>
      <th>card</th>
      <th>away_fault</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13327</td>
      <td>90+3</td>
      <td>Can E.</td>
      <td>yellow card</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13329</td>
      <td>29</td>
      <td>Holebas J.</td>
      <td>yellow card</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13329</td>
      <td>40</td>
      <td>Doucoure A.</td>
      <td>yellow card</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13331</td>
      <td>33</td>
      <td>Cabaye Y.</td>
      <td>yellow card</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13331</td>
      <td>50</td>
      <td>Puncheon J.</td>
      <td>yellow card</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
bets = pd.read_csv("bets.csv")
bets.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>odd_bookmakers</th>
      <th>odd_epoch</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>146845</td>
      <td>BetOlimp</td>
      <td>1486301854</td>
      <td>odd_1</td>
      <td>1.96</td>
    </tr>
    <tr>
      <th>1</th>
      <td>151780</td>
      <td>10Bet</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>151780</td>
      <td>18bet</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151780</td>
      <td>1xBet</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>151780</td>
      <td>5Dimes</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.23</td>
    </tr>
  </tbody>
</table>
</div>



**odd_x : Draw**

**odd_1 : Home Wins**

**odd_2 : Away Wins**

Let's create another column which contains the probabilities that has been given by bookmakers using <br><br> $$\frac {1}{P(value)}$$


```python
prob = list()
for i in bets.value:
    try:
        temp = 1/i
        prob.append(temp)
    except:
        temp = 0
        prob.append(temp)

bets["possibility"] = prob
```


```python
bets["possibility"].head()
```




    0    0.510204
    1    0.465116
    2    0.460829
    3    0.454545
    4    0.448430
    Name: possibility, dtype: float64



### Lets create another column into our matches data to see if a match ended in a draw or no:


```python
data["results"] =  data["match_awayteam_score"] == data["match_hometeam_score"]
data.results.head()
```




    1     False
    5     False
    6      True
    7     False
    27     True
    Name: results, dtype: bool




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_awayteam_id</th>
      <th>match_hometeam_id</th>
      <th>match_id</th>
      <th>epoch</th>
      <th>match_status</th>
      <th>match_live</th>
      <th>match_hometeam_name</th>
      <th>match_awayteam_name</th>
      <th>match_hometeam_score</th>
      <th>match_awayteam_score</th>
      <th>match_hometeam_halftime_score</th>
      <th>match_awayteam_halftime_score</th>
      <th>match_hometeam_extra_score</th>
      <th>match_awayteam_extra_score</th>
      <th>match_hometeam_penalty_score</th>
      <th>match_awayteam_penalty_score</th>
      <th>league_id</th>
      <th>results</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2614</td>
      <td>2619</td>
      <td>13331</td>
      <td>1505561400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Crystal Palace</td>
      <td>Southampton</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2626</td>
      <td>2623</td>
      <td>13329</td>
      <td>1505570400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Watford</td>
      <td>Manchester City</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2629</td>
      <td>2621</td>
      <td>13327</td>
      <td>1505570400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Liverpool</td>
      <td>Burnley</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2641</td>
      <td>2654</td>
      <td>13456</td>
      <td>1505570400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Sheffield Utd</td>
      <td>Norwich</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2617</td>
      <td>2616</td>
      <td>13324</td>
      <td>1505651400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Chelsea</td>
      <td>Arsenal</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
bets.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>odd_bookmakers</th>
      <th>odd_epoch</th>
      <th>variable</th>
      <th>value</th>
      <th>possibility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>146845</td>
      <td>BetOlimp</td>
      <td>1486301854</td>
      <td>odd_1</td>
      <td>1.96</td>
      <td>0.510204</td>
    </tr>
    <tr>
      <th>1</th>
      <td>151780</td>
      <td>10Bet</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.15</td>
      <td>0.465116</td>
    </tr>
    <tr>
      <th>2</th>
      <td>151780</td>
      <td>18bet</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.17</td>
      <td>0.460829</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151780</td>
      <td>1xBet</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.20</td>
      <td>0.454545</td>
    </tr>
    <tr>
      <th>4</th>
      <td>151780</td>
      <td>5Dimes</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.23</td>
      <td>0.448430</td>
    </tr>
  </tbody>
</table>
</div>



We filter the Bets data to get only draw bets. And add a new column named **draw** to which contain a boolean data according to the match result, Draw or No. We make it all False for now.


```python
x = bets['variable'] == 'odd_x'
bets["draw"] = False
bets_draw = bets[x]
```


```python
bets_draw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>odd_bookmakers</th>
      <th>odd_epoch</th>
      <th>variable</th>
      <th>value</th>
      <th>possibility</th>
      <th>draw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>146845</td>
      <td>BetOlimp</td>
      <td>1486301854</td>
      <td>odd_x</td>
      <td>3.49</td>
      <td>0.286533</td>
      <td>False</td>
    </tr>
    <tr>
      <th>36</th>
      <td>151780</td>
      <td>10Bet</td>
      <td>1486314920</td>
      <td>odd_x</td>
      <td>3.15</td>
      <td>0.317460</td>
      <td>False</td>
    </tr>
    <tr>
      <th>37</th>
      <td>151780</td>
      <td>18bet</td>
      <td>1486314920</td>
      <td>odd_x</td>
      <td>3.12</td>
      <td>0.320513</td>
      <td>False</td>
    </tr>
    <tr>
      <th>38</th>
      <td>151780</td>
      <td>1xBet</td>
      <td>1486314920</td>
      <td>odd_x</td>
      <td>3.24</td>
      <td>0.308642</td>
      <td>False</td>
    </tr>
    <tr>
      <th>39</th>
      <td>151780</td>
      <td>5Dimes</td>
      <td>1486314920</td>
      <td>odd_x</td>
      <td>3.19</td>
      <td>0.313480</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.results.value_counts()
```




    False    493
    True     135
    Name: results, dtype: int64




```python
count = int()
for index, row in bets_draw.iterrows():
    id = row["match_id"]
    match = data.query(f'match_id == {id}')
    if len(match) > 0:
        if match.results.values[0]:
            count+=1
            bets_draw.at[index,'draw'] = True
```


```python
count
```




    3783




```python
bets_draw.draw.value_counts()
```




    False    140999
    True       3783
    Name: draw, dtype: int64




```python
%%capture
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()
```


```python
indexes = list()
with tqdm(total = len(bets_draw)) as pbar:
    for index,row in bets_draw.iterrows():
        id = row['match_id']
        match = data.query(f'match_id == {id}')
        if len(match) == 0:
            indexes.append(index)
        elif len(match) > 0:
            bets_draw.at[index,'draw'] = True
        pbar.update(1)

bets_draw = bets_draw.drop(indexes)
```


    HBox(children=(IntProgress(value=0, max=144782), HTML(value='')))


    
    


```python
bets_draw.draw.value_counts()
```




    True    19660
    Name: draw, dtype: int64



We put 4 different dataset of draw bets from 4 different Bookmakers into a python list named **makers**


```python
bookmakers = bets[x].odd_bookmakers.unique().tolist()  # Get a list of bookmakers.
#type(bookmakers)

makers = list()

for i in bookmakers[:4]:  #For the first 4 bookmakers in the list bookmakers. We get their data into makers list.
    bookmaker = bets[x]['odd_bookmakers'] == i
    
    makers.append(bets[x][bookmaker])
```

Lets see the first element of the list makers to have a better understanding of what we done.

First element of makers list contains data of BetOlimp bookmakers draw data.


```python
makers[0].head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>odd_bookmakers</th>
      <th>odd_epoch</th>
      <th>variable</th>
      <th>value</th>
      <th>possibility</th>
      <th>draw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>146845</td>
      <td>BetOlimp</td>
      <td>1486301854</td>
      <td>odd_x</td>
      <td>3.49</td>
      <td>0.286533</td>
      <td>False</td>
    </tr>
    <tr>
      <th>42</th>
      <td>151780</td>
      <td>BetOlimp</td>
      <td>1486314920</td>
      <td>odd_x</td>
      <td>3.15</td>
      <td>0.317460</td>
      <td>False</td>
    </tr>
    <tr>
      <th>59</th>
      <td>151781</td>
      <td>BetOlimp</td>
      <td>1486314941</td>
      <td>odd_x</td>
      <td>3.28</td>
      <td>0.304878</td>
      <td>False</td>
    </tr>
    <tr>
      <th>166</th>
      <td>147990</td>
      <td>BetOlimp</td>
      <td>1486710451</td>
      <td>odd_x</td>
      <td>3.23</td>
      <td>0.309598</td>
      <td>False</td>
    </tr>
    <tr>
      <th>183</th>
      <td>148000</td>
      <td>BetOlimp</td>
      <td>1486710493</td>
      <td>odd_x</td>
      <td>2.99</td>
      <td>0.334448</td>
      <td>False</td>
    </tr>
    <tr>
      <th>200</th>
      <td>148001</td>
      <td>BetOlimp</td>
      <td>1486710502</td>
      <td>odd_x</td>
      <td>3.22</td>
      <td>0.310559</td>
      <td>False</td>
    </tr>
    <tr>
      <th>378</th>
      <td>147991</td>
      <td>BetOlimp</td>
      <td>1486797015</td>
      <td>odd_x</td>
      <td>3.51</td>
      <td>0.284900</td>
      <td>False</td>
    </tr>
    <tr>
      <th>395</th>
      <td>147992</td>
      <td>BetOlimp</td>
      <td>1486797022</td>
      <td>odd_x</td>
      <td>5.03</td>
      <td>0.198807</td>
      <td>False</td>
    </tr>
    <tr>
      <th>412</th>
      <td>147993</td>
      <td>BetOlimp</td>
      <td>1486797032</td>
      <td>odd_x</td>
      <td>4.59</td>
      <td>0.217865</td>
      <td>False</td>
    </tr>
    <tr>
      <th>429</th>
      <td>147995</td>
      <td>BetOlimp</td>
      <td>1486710471</td>
      <td>odd_x</td>
      <td>3.54</td>
      <td>0.282486</td>
      <td>False</td>
    </tr>
    <tr>
      <th>446</th>
      <td>147996</td>
      <td>BetOlimp</td>
      <td>1486797049</td>
      <td>odd_x</td>
      <td>3.60</td>
      <td>0.277778</td>
      <td>False</td>
    </tr>
    <tr>
      <th>612</th>
      <td>147997</td>
      <td>BetOlimp</td>
      <td>1486797054</td>
      <td>odd_x</td>
      <td>3.45</td>
      <td>0.289855</td>
      <td>False</td>
    </tr>
    <tr>
      <th>629</th>
      <td>147998</td>
      <td>BetOlimp</td>
      <td>1486929695</td>
      <td>odd_x</td>
      <td>3.21</td>
      <td>0.311526</td>
      <td>False</td>
    </tr>
    <tr>
      <th>646</th>
      <td>154303</td>
      <td>BetOlimp</td>
      <td>1486931765</td>
      <td>odd_x</td>
      <td>3.68</td>
      <td>0.271739</td>
      <td>False</td>
    </tr>
    <tr>
      <th>752</th>
      <td>150848</td>
      <td>BetOlimp</td>
      <td>1487365197</td>
      <td>odd_x</td>
      <td>3.16</td>
      <td>0.316456</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



When we need to check if a match ended in a draw or no from its match_id it is more simple to have a dataset like below:


```python
match_results = data.iloc[:,[2,-1]]
match_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>results</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41196</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13331</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17683</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17684</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17682</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### 4 Bookmakers that we are Going to Analyse their Draw Possiblities


```python
for i in makers:
    print(i.odd_bookmakers.unique())
    continue
```

    ['BetOlimp']
    ['10Bet']
    ['18bet']
    ['1xBet']
    

Now let's turn the "draw" column in each element of makers list which belongs to different bookamkers to True or False if correnponding match ended in draw.

To do that we need to check match ID's from the small dataset we created called match_results


```python
for index,rows in makers[0].iterrows():
    id = rows['match_id']
    if match_results.query(f'match_id == {id}').results.values[0]:
        makers[0].at[index,'draw'] = True
```


```python
for index,rows in makers[1].iterrows():
    id = rows['match_id']
    if match_results.query(f"match_id == {id}").results.values[0]:
        makers[1].at[index,'draw'] = True
```


```python
for index,rows in makers[2].iterrows():
    id = rows['match_id']
    if match_results.query(f"match_id == {id}").results.values[0]:
        makers[2].at[index,'draw'] = True
```


```python
for index,rows in makers[3].iterrows():
    id = rows['match_id']
    if match_results.query(f"match_id == {id}").results.values[0]:
        makers[3].at[index,'draw'] = True
```

### Splitting Probabilities into Bins for the Bookmaker BetOlimp

Now we split datasets into bins by their probability of draw.

### **Probabibility Range of Bins**

- ( 0.0 - 0.10 ]
- ( 0.10 - 0.20 ]
- ( 0.20 - 0.30 ]
- ( 0.30 - 0.40 ]


```python
bins_for_betolimp = list()
```


```python
x = makers[0][np.logical_and(makers[0]['possibility'] > 0.0 , makers[0]['possibility'] <= 0.10)]
x2 = makers[0][np.logical_and(makers[0]['possibility'] > 0.10 , makers[0]['possibility'] <= 0.20)]
x3 = makers[0][np.logical_and(makers[0]['possibility'] > 0.20 , makers[0]['possibility'] <= 0.30)]
x4 = makers[0][np.logical_and(makers[0]['possibility'] > 0.30 , makers[0]['possibility'] <= 0.40)]


x5 = makers[0][np.logical_and(makers[0]['possibility'] > 0.40 , makers[0]['possibility'] <= 0.50)]
x6 = makers[0][np.logical_and(makers[0]['possibility'] > 0.50 , makers[0]['possibility'] <= 0.60)]
x7 = makers[0][np.logical_and(makers[0]['possibility'] > 0.60 , makers[0]['possibility'] <= 0.70)]
x8 = makers[0][np.logical_and(makers[0]['possibility'] > 0.70 , makers[0]['possibility'] <= 0.80)]
```

Note that there are no probability of draw biger than 0.40 for some datasets.


```python
bins_for_betolimp = list()
bins_for_betolimp.append(x)
bins_for_betolimp.append(x2)
bins_for_betolimp.append(x3)
bins_for_betolimp.append(x4)
bins_for_betolimp.append(x5)
```


```python
x = makers[1][np.logical_and(makers[1]['possibility'] > 0.0 , makers[1]['possibility'] <= 0.10)]
x2 = makers[1][np.logical_and(makers[1]['possibility'] > 0.10 , makers[1]['possibility'] <= 0.20)]
x3 = makers[1][np.logical_and(makers[1]['possibility'] > 0.20 , makers[1]['possibility'] <= 0.30)]
x4 = makers[1][np.logical_and(makers[1]['possibility'] > 0.30 , makers[1]['possibility'] <= 0.40)]


x5 = makers[1][np.logical_and(makers[1]['possibility'] > 0.40 , makers[1]['possibility'] <= 0.50)]
x6 = makers[1][np.logical_and(makers[1]['possibility'] > 0.50 , makers[1]['possibility'] <= 0.60)]
x7 = makers[1][np.logical_and(makers[1]['possibility'] > 0.60 , makers[1]['possibility'] <= 0.70)]
x8 = makers[1][np.logical_and(makers[1]['possibility'] > 0.70 , makers[1]['possibility'] <= 0.80)]
```


```python
bins_for_10bet = list()

bins_for_10bet.append(x)
bins_for_10bet.append(x2)
bins_for_10bet.append(x3)
bins_for_10bet.append(x4)
bins_for_10bet.append(x5)
```


```python
x = makers[2][np.logical_and(makers[2]['possibility'] > 0.0 , makers[2]['possibility'] <= 0.10)]
x2 = makers[2][np.logical_and(makers[2]['possibility'] > 0.10 , makers[2]['possibility'] <= 0.20)]
x3 = makers[2][np.logical_and(makers[2]['possibility'] > 0.20 , makers[2]['possibility'] <= 0.30)]
x4 = makers[2][np.logical_and(makers[2]['possibility'] > 0.30 , makers[2]['possibility'] <= 0.40)]


x5 = makers[2][np.logical_and(makers[2]['possibility'] > 0.40 , makers[2]['possibility'] <= 0.50)]
x6 = makers[2][np.logical_and(makers[2]['possibility'] > 0.50 , makers[2]['possibility'] <= 0.60)]
x7 = makers[2][np.logical_and(makers[2]['possibility'] > 0.60 , makers[2]['possibility'] <= 0.70)]
x8 = makers[2][np.logical_and(makers[2]['possibility'] > 0.70 , makers[2]['possibility'] <= 0.80)]
```


```python
bins_for_18bet = list()

bins_for_18bet.append(x)
bins_for_18bet.append(x2)
bins_for_18bet.append(x3)
bins_for_18bet.append(x4)
bins_for_18bet.append(x5)
```


```python
x = makers[3][np.logical_and(makers[3]['possibility'] > 0.0 , makers[3]['possibility'] <= 0.10)]
x2 = makers[3][np.logical_and(makers[3]['possibility'] > 0.10 , makers[3]['possibility'] <= 0.20)]
x3 = makers[3][np.logical_and(makers[3]['possibility'] > 0.20 , makers[3]['possibility'] <= 0.30)]
x4 = makers[3][np.logical_and(makers[3]['possibility'] > 0.30 , makers[3]['possibility'] <= 0.40)]


x5 = makers[3][np.logical_and(makers[3]['possibility'] > 0.40 , makers[3]['possibility'] <= 0.50)]
x6 = makers[3][np.logical_and(makers[3]['possibility'] > 0.50 , makers[3]['possibility'] <= 0.60)]
x7 = makers[3][np.logical_and(makers[3]['possibility'] > 0.60 , makers[3]['possibility'] <= 0.70)]
x8 = makers[3][np.logical_and(makers[3]['possibility'] > 0.70 , makers[3]['possibility'] <= 0.80)]
```


```python
bins_for_1xbet = list()

bins_for_1xbet.append(x)
bins_for_1xbet.append(x2)
bins_for_1xbet.append(x3)
bins_for_1xbet.append(x4)
bins_for_1xbet.append(x5)
```

### Estimated Probability of Draws For BetOlimp

When we divide the number of matches ended in draw in the $Bin_i$ with the total number of matches in $Bin_i$: <br>

$$Estimated\space Probability\space of\space Draw = \frac{Number\space of\space Matches\space Ended\space In\space Draw\space in\space Bin_i}{Number\space of\space Matches\space In\space Bin_i}$$


```python
print("\nBetOlimp Estimated Probability of Draws\n\n")
print("Bookmakers Probability : Estimated Probability")
for i in bins_for_betolimp:
    k = i.draw == True
    try:
        print("(" + str(round(i.possibility.min(),2)) + " - " + str(round(i.possibility.max(),2)) +  "]        :        " + str(round(len(i[k])/len(i),2)))
    except:
        pass
    
```

    
    BetOlimp Estimated Probability of Draws
    
    
    Bookmakers Probability : Estimated Probability
    (0.07 - 0.1]        :        0.0
    (0.1 - 0.2]        :        0.19
    (0.2 - 0.3]        :        0.24
    (0.3 - 0.4]        :        0.27
    

### Estimated Probability of Draws For 10Bet


```python
print("\n10Bet Estimated Probability of Draws\n\n")
print("Bookmakers Probability : Estimated Probability")
for i in bins_for_10bet:
    k = i.draw == True
    try:
        print("(" + str(round(i.possibility.min(),2)) + " - " + str(round(i.possibility.max(),2)) +  "]        :         " + str(round(len(i[k])/len(i),2)))
    except:
        pass
```

    
    10Bet Estimated Probability of Draws
    
    
    Bookmakers Probability : Estimated Probability
    (0.07 - 0.1]        :         0.09
    (0.1 - 0.2]        :         0.12
    (0.2 - 0.3]        :         0.24
    (0.3 - 0.39]        :         0.28
    

### Estimated Probability of Draws For 18Bet


```python
print("\n18Bet Estimated Probability of Draws\n\n")
print("Bookmakers Probability : Estimated Probability")
for i in bins_for_18bet:
    k = i.draw == True
    try:
        print("(" + str(round(i.possibility.min(),2)) + " - " + str(round(i.possibility.max(),2)) +  "]        :         " + str(round(len(i[k])/len(i),2)))
    except:
        pass
```

    
    18Bet Estimated Probability of Draws
    
    
    Bookmakers Probability : Estimated Probability
    (0.07 - 0.1]        :         0.1
    (0.11 - 0.2]        :         0.13
    (0.2 - 0.3]        :         0.24
    (0.3 - 0.4]        :         0.29
    

### Estimated Probability of Draws For 1xBet


```python
print("\n1xBet Estimated Probability of Draws\n\n")
print("Bookmakers Probability : Estimated Probability")
for i in bins_for_1xbet:
    k = i.draw == True
    try:
        print("(" + str(round(i.possibility.min(),2)) + " - " + str(round(i.possibility.max(),2)) +  "]        :         " + str(round(len(i[k])/len(i),2)))
    except:
        pass
```

    
    1xBet Estimated Probability of Draws
    
    
    Bookmakers Probability : Estimated Probability
    (0.07 - 0.1]        :         0.04
    (0.1 - 0.2]        :         0.16
    (0.2 - 0.3]        :         0.24
    (0.3 - 0.39]        :         0.3
    (0.4 - 0.41]        :         0.0
    

Since we are done with the draw boolen column, we can dop it:


```python
bets = bets.drop(['draw'], axis=1)
```

Now let's take a look at the home win bets:


```python
#bets = bets.drop(['away_win'], axis=1)
x = bets['variable'] == 'odd_1'
bets["home_win"] = False
bets[x].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>odd_bookmakers</th>
      <th>odd_epoch</th>
      <th>variable</th>
      <th>value</th>
      <th>possibility</th>
      <th>home_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>146845</td>
      <td>BetOlimp</td>
      <td>1486301854</td>
      <td>odd_1</td>
      <td>1.96</td>
      <td>0.510204</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>151780</td>
      <td>10Bet</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.15</td>
      <td>0.465116</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>151780</td>
      <td>18bet</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.17</td>
      <td>0.460829</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151780</td>
      <td>1xBet</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.20</td>
      <td>0.454545</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>151780</td>
      <td>5Dimes</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.23</td>
      <td>0.448430</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>151780</td>
      <td>bet-at-home</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.12</td>
      <td>0.471698</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>151780</td>
      <td>bet365</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.20</td>
      <td>0.454545</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>151780</td>
      <td>BetOlimp</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.19</td>
      <td>0.456621</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>151780</td>
      <td>Betrally</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.15</td>
      <td>0.465116</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>151780</td>
      <td>BetVictor</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.20</td>
      <td>0.454545</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
bookmakers = bets[x].odd_bookmakers.unique().tolist()
type(bookmakers)

makers_1 = list()

for i in bookmakers[:4]:
    bookmaker = bets[x]['odd_bookmakers'] == i
    
    makers_1.append(bets[x][bookmaker])
```


```python
bets.query(f'match_id == 146845')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>odd_bookmakers</th>
      <th>odd_epoch</th>
      <th>variable</th>
      <th>value</th>
      <th>possibility</th>
      <th>home_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>146845</td>
      <td>BetOlimp</td>
      <td>1486301854</td>
      <td>odd_1</td>
      <td>1.96</td>
      <td>0.510204</td>
      <td>False</td>
    </tr>
    <tr>
      <th>35</th>
      <td>146845</td>
      <td>BetOlimp</td>
      <td>1486301854</td>
      <td>odd_x</td>
      <td>3.49</td>
      <td>0.286533</td>
      <td>False</td>
    </tr>
    <tr>
      <th>70</th>
      <td>146845</td>
      <td>BetOlimp</td>
      <td>1486301854</td>
      <td>odd_2</td>
      <td>3.65</td>
      <td>0.273973</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
makers_1[0].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>odd_bookmakers</th>
      <th>odd_epoch</th>
      <th>variable</th>
      <th>value</th>
      <th>possibility</th>
      <th>home_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>146845</td>
      <td>BetOlimp</td>
      <td>1486301854</td>
      <td>odd_1</td>
      <td>1.96</td>
      <td>0.510204</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>151780</td>
      <td>BetOlimp</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.19</td>
      <td>0.456621</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>151781</td>
      <td>BetOlimp</td>
      <td>1486314941</td>
      <td>odd_1</td>
      <td>2.95</td>
      <td>0.338983</td>
      <td>False</td>
    </tr>
    <tr>
      <th>111</th>
      <td>147990</td>
      <td>BetOlimp</td>
      <td>1486710451</td>
      <td>odd_1</td>
      <td>2.14</td>
      <td>0.467290</td>
      <td>False</td>
    </tr>
    <tr>
      <th>128</th>
      <td>148000</td>
      <td>BetOlimp</td>
      <td>1486710493</td>
      <td>odd_1</td>
      <td>2.50</td>
      <td>0.400000</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_awayteam_id</th>
      <th>match_hometeam_id</th>
      <th>match_id</th>
      <th>epoch</th>
      <th>match_status</th>
      <th>match_live</th>
      <th>match_hometeam_name</th>
      <th>match_awayteam_name</th>
      <th>match_hometeam_score</th>
      <th>match_awayteam_score</th>
      <th>match_hometeam_halftime_score</th>
      <th>match_awayteam_halftime_score</th>
      <th>match_hometeam_extra_score</th>
      <th>match_awayteam_extra_score</th>
      <th>match_hometeam_penalty_score</th>
      <th>match_awayteam_penalty_score</th>
      <th>league_id</th>
      <th>results</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7109</td>
      <td>7097</td>
      <td>41196</td>
      <td>1505559600</td>
      <td>Finished</td>
      <td>0</td>
      <td>Levante</td>
      <td>Valencia</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>468</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2614</td>
      <td>2619</td>
      <td>13331</td>
      <td>1505561400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Crystal Palace</td>
      <td>Southampton</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3224</td>
      <td>3238</td>
      <td>17683</td>
      <td>1505568600</td>
      <td>Finished</td>
      <td>0</td>
      <td>Eintracht Frankfurt</td>
      <td>FC Augsburg</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>195</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3235</td>
      <td>3223</td>
      <td>17684</td>
      <td>1505568600</td>
      <td>Finished</td>
      <td>0</td>
      <td>SV Werder Bremen</td>
      <td>Schalke</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>195</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3237</td>
      <td>3225</td>
      <td>17682</td>
      <td>1505568600</td>
      <td>Finished</td>
      <td>0</td>
      <td>Bayern Munich</td>
      <td>1. FSV Mainz 05</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>195</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



We extract the part of the data which has home team wins. Since we dont have a column that tells us the result, we compare goals of home teams and away teams and add the matches with bigger home team goals to the new dataset called **home_wins**


```python
home_wins = data[data['match_hometeam_score'] > data['match_awayteam_score']]
home_wins.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_awayteam_id</th>
      <th>match_hometeam_id</th>
      <th>match_id</th>
      <th>epoch</th>
      <th>match_status</th>
      <th>match_live</th>
      <th>match_hometeam_name</th>
      <th>match_awayteam_name</th>
      <th>match_hometeam_score</th>
      <th>match_awayteam_score</th>
      <th>match_hometeam_halftime_score</th>
      <th>match_awayteam_halftime_score</th>
      <th>match_hometeam_extra_score</th>
      <th>match_awayteam_extra_score</th>
      <th>match_hometeam_penalty_score</th>
      <th>match_awayteam_penalty_score</th>
      <th>league_id</th>
      <th>results</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>3237</td>
      <td>3225</td>
      <td>17682</td>
      <td>1505568600</td>
      <td>Finished</td>
      <td>0</td>
      <td>Bayern Munich</td>
      <td>1. FSV Mainz 05</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>195</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2642</td>
      <td>2650</td>
      <td>13448</td>
      <td>1505570400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Bristol City</td>
      <td>Derby</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>149</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2645</td>
      <td>2648</td>
      <td>13453</td>
      <td>1505570400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Middlesbrough</td>
      <td>QPR</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>149</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2653</td>
      <td>2638</td>
      <td>13454</td>
      <td>1505570400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Millwall</td>
      <td>Leeds</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>149</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>7110</td>
      <td>7115</td>
      <td>41313</td>
      <td>1505570400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Valladolid</td>
      <td>Granada CF</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>468</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in makers_1:
    for index,rows in i.iterrows():
        id = rows['match_id']
        if len(home_wins.query(f'match_id == {id}')) > 0:
            i.at[index,'home_win'] = True
```


```python
bins_for_10bet_1 = list()
bins_for_18bet_1 = list()
bins_for_betolimp_1 = list()
bins_for_1xbet_1 = list()

for i in makers_1:
    poss = 0.0
    if i['odd_bookmakers'].unique().tolist()[0] == 'BetOlimp':
        while poss <= 0.90:
            bins_for_betolimp_1.append(i[np.logical_and(i['possibility'] > poss , i['possibility'] <= (poss + 0.10))])
            poss += 0.10
    poss = 0.0
    if i['odd_bookmakers'].unique().tolist()[0] == '10Bet':
        while poss <= 0.90:
            bins_for_10bet_1.append(i[np.logical_and(i['possibility'] > poss , i['possibility'] <= (poss + 0.10))])
            poss += 0.10
    poss = 0.0
    if i['odd_bookmakers'].unique().tolist()[0] == '18Bet':
        while poss <= 0.90:
            bins_for_18bet_1.append(i[np.logical_and(i['possibility'] > poss , i['possibility'] <= (poss + 0.10))])
            poss += 0.10
    poss = 0.0
    if i['odd_bookmakers'].unique().tolist()[0] == '1xBet':
        while poss <= 0.90:
            bins_for_10bet_1.append(i[np.logical_and(i['possibility'] > poss , i['possibility'] <= (poss + 0.10))])
            poss += 0.10
    poss = 0.0
```


```python
for i in makers_1:
    print(i['odd_bookmakers'].unique().tolist())
    
```

    ['BetOlimp']
    ['10Bet']
    ['18bet']
    ['1xBet']
    


```python
#bets = bets.drop(['home_win'], axis = 1)
x = bets['variable'] == 'odd_2'
bets["away_win"] = False
bets[x].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>odd_bookmakers</th>
      <th>odd_epoch</th>
      <th>variable</th>
      <th>value</th>
      <th>possibility</th>
      <th>away_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>70</th>
      <td>146845</td>
      <td>BetOlimp</td>
      <td>1486301854</td>
      <td>odd_2</td>
      <td>3.65</td>
      <td>0.273973</td>
      <td>False</td>
    </tr>
    <tr>
      <th>71</th>
      <td>151780</td>
      <td>10Bet</td>
      <td>1486314920</td>
      <td>odd_2</td>
      <td>3.70</td>
      <td>0.270270</td>
      <td>False</td>
    </tr>
    <tr>
      <th>72</th>
      <td>151780</td>
      <td>18bet</td>
      <td>1486314920</td>
      <td>odd_2</td>
      <td>3.32</td>
      <td>0.301205</td>
      <td>False</td>
    </tr>
    <tr>
      <th>73</th>
      <td>151780</td>
      <td>1xBet</td>
      <td>1486314920</td>
      <td>odd_2</td>
      <td>3.88</td>
      <td>0.257732</td>
      <td>False</td>
    </tr>
    <tr>
      <th>74</th>
      <td>151780</td>
      <td>5Dimes</td>
      <td>1486314920</td>
      <td>odd_2</td>
      <td>3.82</td>
      <td>0.261780</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
bookmakers = bets[x].odd_bookmakers.unique().tolist()
type(bookmakers)

makers_2 = list()

for i in bookmakers[:4]:
    bookmaker = bets[x]['odd_bookmakers'] == i
    
    makers_2.append(bets[x][bookmaker])
```


```python
away_wins = data[data['match_hometeam_score'] < data['match_awayteam_score']]
away_wins.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_awayteam_id</th>
      <th>match_hometeam_id</th>
      <th>match_id</th>
      <th>epoch</th>
      <th>match_status</th>
      <th>match_live</th>
      <th>match_hometeam_name</th>
      <th>match_awayteam_name</th>
      <th>match_hometeam_score</th>
      <th>match_awayteam_score</th>
      <th>match_hometeam_halftime_score</th>
      <th>match_awayteam_halftime_score</th>
      <th>match_hometeam_extra_score</th>
      <th>match_awayteam_extra_score</th>
      <th>match_hometeam_penalty_score</th>
      <th>match_awayteam_penalty_score</th>
      <th>league_id</th>
      <th>results</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2614</td>
      <td>2619</td>
      <td>13331</td>
      <td>1505561400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Crystal Palace</td>
      <td>Southampton</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3224</td>
      <td>3238</td>
      <td>17683</td>
      <td>1505568600</td>
      <td>Finished</td>
      <td>0</td>
      <td>Eintracht Frankfurt</td>
      <td>FC Augsburg</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>195</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3235</td>
      <td>3223</td>
      <td>17684</td>
      <td>1505568600</td>
      <td>Finished</td>
      <td>0</td>
      <td>SV Werder Bremen</td>
      <td>Schalke</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>195</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2626</td>
      <td>2623</td>
      <td>13329</td>
      <td>1505570400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Watford</td>
      <td>Manchester City</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2641</td>
      <td>2654</td>
      <td>13456</td>
      <td>1505570400</td>
      <td>Finished</td>
      <td>0</td>
      <td>Sheffield Utd</td>
      <td>Norwich</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in makers_2:
    for index,rows in i.iterrows():
        id = rows['match_id']
        if len(away_wins.query(f'match_id == {id}')) > 0:
            i.at[index,'away_win'] = True
```


```python
bins_for_10bet_2 = list()
bins_for_18bet_2 = list()
bins_for_betolimp_2 = list()
bins_for_1xbet_2 = list()

for i in makers_2:
    poss = 0.0
    if i['odd_bookmakers'].unique().tolist()[0] == 'BetOlimp':
        while poss <= 0.90:
            bins_for_betolimp_2.append(i[np.logical_and(i['possibility'] > poss , i['possibility'] <= (poss + 0.10))])
            poss += 0.10
    poss = 0.0
    if i['odd_bookmakers'].unique().tolist()[0] == '10Bet':
        while poss <= 0.90:
            bins_for_10bet_2.append(i[np.logical_and(i['possibility'] > poss , i['possibility'] <= (poss + 0.10))])
            poss += 0.10
    poss = 0.0
    if i['odd_bookmakers'].unique().tolist()[0] == '18Bet':
        while poss <= 0.90:
            bins_for_18bet_2.append(i[np.logical_and(i['possibility'] > poss , i['possibility'] <= (poss + 0.10))])
            poss += 0.10
    poss = 0.0
    if i['odd_bookmakers'].unique().tolist()[0] == '1xBet':
        while poss <= 0.90:
            bins_for_10bet_2.append(i[np.logical_and(i['possibility'] > poss , i['possibility'] <= (poss + 0.10))])
            poss += 0.10
    poss = 0.0
```


```python
makers_1[0].head()
#makers[0].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>odd_bookmakers</th>
      <th>odd_epoch</th>
      <th>variable</th>
      <th>value</th>
      <th>possibility</th>
      <th>home_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>146845</td>
      <td>BetOlimp</td>
      <td>1486301854</td>
      <td>odd_1</td>
      <td>1.96</td>
      <td>0.510204</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>151780</td>
      <td>BetOlimp</td>
      <td>1486314920</td>
      <td>odd_1</td>
      <td>2.19</td>
      <td>0.456621</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>151781</td>
      <td>BetOlimp</td>
      <td>1486314941</td>
      <td>odd_1</td>
      <td>2.95</td>
      <td>0.338983</td>
      <td>True</td>
    </tr>
    <tr>
      <th>111</th>
      <td>147990</td>
      <td>BetOlimp</td>
      <td>1486710451</td>
      <td>odd_1</td>
      <td>2.14</td>
      <td>0.467290</td>
      <td>False</td>
    </tr>
    <tr>
      <th>128</th>
      <td>148000</td>
      <td>BetOlimp</td>
      <td>1486710493</td>
      <td>odd_1</td>
      <td>2.50</td>
      <td>0.400000</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
home_away = pd.concat([(-makers_2[0]["possibility"]), makers_1[0]["possibility"]])
home_away.head()
                
```




    70    -0.273973
    77    -0.275482
    94    -0.406504
    221   -0.263852
    238   -0.335570
    Name: possibility, dtype: float64




```python


plt.figure(figsize= (16,7))

plt.subplot(1,2,1)
plt.title("BetOlimp")
plt.scatter(makers_1[0]["possibility"], makers[0]["possibility"], alpha = 0.05)
plt.xlabel("Home-Win Possibilities")
plt.ylabel("Draw Possibilities")
plt.subplot(1,2,2)
plt.title("10Bet")
plt.scatter(makers_1[1]["possibility"], makers[1]["possibility"], alpha = 0.05)
plt.xlabel("Home-Win Possibilities")
plt.ylabel("Draw Possibilities")
plt.show()
```


![png](output_97_0.png)


# Homework 2


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
```

## Task 1

Column 1 --> Bag Class

Column 2 --> Bag ID


```python
data = pd.read_csv('Musk1.csv', header = None)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>158</th>
      <th>159</th>
      <th>160</th>
      <th>161</th>
      <th>162</th>
      <th>163</th>
      <th>164</th>
      <th>165</th>
      <th>166</th>
      <th>167</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>-198</td>
      <td>-109</td>
      <td>-75</td>
      <td>-117</td>
      <td>11</td>
      <td>23</td>
      <td>-88</td>
      <td>...</td>
      <td>-238</td>
      <td>-74</td>
      <td>-129</td>
      <td>-120</td>
      <td>-38</td>
      <td>30</td>
      <td>48</td>
      <td>-37</td>
      <td>6</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>-191</td>
      <td>-142</td>
      <td>-65</td>
      <td>-117</td>
      <td>55</td>
      <td>49</td>
      <td>-170</td>
      <td>...</td>
      <td>-238</td>
      <td>-302</td>
      <td>60</td>
      <td>-120</td>
      <td>-39</td>
      <td>31</td>
      <td>48</td>
      <td>-37</td>
      <td>5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>-191</td>
      <td>-142</td>
      <td>-75</td>
      <td>-117</td>
      <td>11</td>
      <td>49</td>
      <td>-161</td>
      <td>...</td>
      <td>-238</td>
      <td>-73</td>
      <td>-127</td>
      <td>-120</td>
      <td>-38</td>
      <td>30</td>
      <td>48</td>
      <td>-37</td>
      <td>5</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>42</td>
      <td>-198</td>
      <td>-110</td>
      <td>-65</td>
      <td>-117</td>
      <td>55</td>
      <td>23</td>
      <td>-95</td>
      <td>...</td>
      <td>-238</td>
      <td>-302</td>
      <td>60</td>
      <td>-120</td>
      <td>-39</td>
      <td>30</td>
      <td>48</td>
      <td>-37</td>
      <td>6</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
      <td>42</td>
      <td>-198</td>
      <td>-102</td>
      <td>-75</td>
      <td>-117</td>
      <td>10</td>
      <td>24</td>
      <td>-87</td>
      <td>...</td>
      <td>-238</td>
      <td>-73</td>
      <td>-127</td>
      <td>51</td>
      <td>128</td>
      <td>144</td>
      <td>43</td>
      <td>-30</td>
      <td>14</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 168 columns</p>
</div>




```python
from sklearn.preprocessing import StandardScaler
x = data.loc[:,2:167]
y = data.loc[:,0]
```

### Let's separate the independent features and dependent class column:


```python
x.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>...</th>
      <th>158</th>
      <th>159</th>
      <th>160</th>
      <th>161</th>
      <th>162</th>
      <th>163</th>
      <th>164</th>
      <th>165</th>
      <th>166</th>
      <th>167</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>-198</td>
      <td>-109</td>
      <td>-75</td>
      <td>-117</td>
      <td>11</td>
      <td>23</td>
      <td>-88</td>
      <td>-28</td>
      <td>-27</td>
      <td>...</td>
      <td>-238</td>
      <td>-74</td>
      <td>-129</td>
      <td>-120</td>
      <td>-38</td>
      <td>30</td>
      <td>48</td>
      <td>-37</td>
      <td>6</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>-191</td>
      <td>-142</td>
      <td>-65</td>
      <td>-117</td>
      <td>55</td>
      <td>49</td>
      <td>-170</td>
      <td>-45</td>
      <td>5</td>
      <td>...</td>
      <td>-238</td>
      <td>-302</td>
      <td>60</td>
      <td>-120</td>
      <td>-39</td>
      <td>31</td>
      <td>48</td>
      <td>-37</td>
      <td>5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>-191</td>
      <td>-142</td>
      <td>-75</td>
      <td>-117</td>
      <td>11</td>
      <td>49</td>
      <td>-161</td>
      <td>-45</td>
      <td>-28</td>
      <td>...</td>
      <td>-238</td>
      <td>-73</td>
      <td>-127</td>
      <td>-120</td>
      <td>-38</td>
      <td>30</td>
      <td>48</td>
      <td>-37</td>
      <td>5</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
      <td>-198</td>
      <td>-110</td>
      <td>-65</td>
      <td>-117</td>
      <td>55</td>
      <td>23</td>
      <td>-95</td>
      <td>-28</td>
      <td>5</td>
      <td>...</td>
      <td>-238</td>
      <td>-302</td>
      <td>60</td>
      <td>-120</td>
      <td>-39</td>
      <td>30</td>
      <td>48</td>
      <td>-37</td>
      <td>6</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>-198</td>
      <td>-102</td>
      <td>-75</td>
      <td>-117</td>
      <td>10</td>
      <td>24</td>
      <td>-87</td>
      <td>-28</td>
      <td>-28</td>
      <td>...</td>
      <td>-238</td>
      <td>-73</td>
      <td>-127</td>
      <td>51</td>
      <td>128</td>
      <td>144</td>
      <td>43</td>
      <td>-30</td>
      <td>14</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 166 columns</p>
</div>




```python
y.head()
```




    0    1
    1    1
    2    1
    3    1
    4    1
    Name: 0, dtype: int64



#### Than we need to standardize these independent features' values:


```python
x = StandardScaler().fit_transform(x)
```


```python
x
```




    array([[ 0.1809131 , -0.88252743, -0.43062779, ..., -0.17946317,
             0.55638614, -0.06861517],
           [ 0.1809131 , -0.80318093, -0.90819777, ..., -0.17946317,
             0.53912936, -0.06861517],
           [ 0.1809131 , -0.80318093, -0.90819777, ..., -0.17946317,
             0.53912936, -0.05014186],
           ...,
           [ 0.2362567 ,  0.20565318,  0.85736399, ..., -0.17946317,
             0.12496664, -1.28785397],
           [ 0.01488231,  0.70440263,  1.53753942, ...,  0.38295103,
             0.5046158 ,  0.74421069],
           [ 0.73434907, -0.0097159 ,  0.79947672, ...,  1.25781757,
             0.65992682,  1.15062363]])



#### Than we import PCA library for python, we choose n = 2 as we want to reduce the number of components to 2:


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
```


```python
principalComponents = pca.fit_transform(x)
```


```python
columns = data[1].unique()
```


```python
principalDf = pd.DataFrame(data = principalComponents
             , columns = ["principal_component_1", "principal_component_2"])
```


```python
finalDf = pd.concat([principalDf, data[0]], axis = 1)
```


```python
finalDf.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>principal_component_1</th>
      <th>principal_component_2</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>471</th>
      <td>-5.253664</td>
      <td>-1.294476</td>
      <td>0</td>
    </tr>
    <tr>
      <th>472</th>
      <td>-6.833537</td>
      <td>-0.135816</td>
      <td>0</td>
    </tr>
    <tr>
      <th>473</th>
      <td>10.587193</td>
      <td>-2.892693</td>
      <td>0</td>
    </tr>
    <tr>
      <th>474</th>
      <td>-4.003161</td>
      <td>-0.307099</td>
      <td>0</td>
    </tr>
    <tr>
      <th>475</th>
      <td>10.507496</td>
      <td>-2.668611</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf[0] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal_component_1']
               , finalDf.loc[indicesToKeep, 'principal_component_2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
```


![png](output_117_0.png)


This seem like, it is not a good transformation approach for classifying bags.

## Task 2

### Let's display our 256x256 photo


```python
image = cv2.imread('selfie.jpg',1)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (6,6))
plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x2398da1f208>




![png](output_121_1.png)



```python
r,g,b = cv2.split(image)
```


```python
noise_R = np.random.uniform(r.min(),r.max()*0.1, r.size)
noise_G = np.random.uniform(g.min(),g.max()*0.1, g.size)
noise_B = np.random.uniform(b.min(),b.max()*0.1, b.size)
```


```python
noise_R = noise_R.reshape(256,256)
noise_G = noise_G.reshape(256,256)
noise_B = noise_B.reshape(256,256)
```


```python
r = r + noise_R
g = g + noise_G
b = b + noise_B
```

Some pixel values could get bigger than 255, to avoid that, we use clip fuction to limit max of a pixel value: 


```python
r = np.clip(r, a_min = 0, a_max = 255)
g = np.clip(g, a_min = 0, a_max = 255)
b = np.clip(b, a_min = 0, a_max = 255)
```


```python
img = cv2.merge((r, g, b))
```

### Noisy Image


```python
plt.figure(figsize = (7, 7))
img = img.astype(np.uint8)
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x23991573668>




![png](output_130_1.png)


### Each channel of the Noisy Image


```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,10))
#fig.suptitle('Channels')
ax1.imshow(r)
ax2.imshow(g)
ax3.imshow(b)
```




    <matplotlib.image.AxesImage at 0x2398ddf77f0>




![png](output_132_1.png)


### Now lets transform our noisy image to grayscale format:


```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
```




    <matplotlib.image.AxesImage at 0x23991626668>




![png](output_134_1.png)



```python
gray
```




    array([[199, 205, 225, ..., 251, 251, 252],
           [197, 205, 216, ..., 252, 252, 252],
           [192, 204, 208, ..., 252, 252, 251],
           ...,
           [ 79,  79,  80, ..., 157, 157, 158],
           [ 81,  79,  78, ..., 158, 156, 168],
           [ 79,  77,  75, ..., 168, 164, 158]], dtype=uint8)



### Import necessary libraries to implement PCA:


```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import scipy.io as sio
import matplotlib.image as image
import pandas as pd
import matplotlib.pyplot as plt
```


```python
gray = normalize(gray)
```


```python
plt.imshow(gray)
```




    <matplotlib.image.AxesImage at 0x23993682160>




![png](output_139_1.png)



```python
pca = PCA(.99)
lower_dimension_data = pca.fit_transform(gray)
```


```python
lower_dimension_data.shape
```




    (256, 45)




```python
approximation = pca.inverse_transform(lower_dimension_data)
approximation.shape
```




    (256, 256)




```python
approximation = approximation.reshape(-1,32,32)
gray = gray.reshape(-1,32,32)
```


```python
for i in range(0,gray.shape[0]):
    gray[i,] = gray[i,].T
    approximation[i,] = approximation[i,].T
    
fig4, axarr = plt.subplots(3,2,figsize=(8,8))
axarr[0,0].imshow(gray[0,],cmap='gray')
axarr[0,0].set_title('Original Image')
axarr[0,0].axis('off')
axarr[0,1].imshow(approximation[0,],cmap='gray')
axarr[0,1].set_title('99% Variation')
axarr[0,1].axis('off')
axarr[1,0].imshow(gray[1,],cmap='gray')
axarr[1,0].set_title('Original Image')
axarr[1,0].axis('off')
axarr[1,1].imshow(approximation[1,],cmap='gray')
axarr[1,1].set_title('99% Variation')
axarr[1,1].axis('off')
axarr[2,0].imshow(gray[2,],cmap='gray')
axarr[2,0].set_title('Original Image')
axarr[2,0].axis('off')
axarr[2,1].imshow(approximation[2,],cmap='gray')
axarr[2,1].set_title('99% variation')
axarr[2,1].axis('off')
plt.show()
```


![png](output_144_0.png)

