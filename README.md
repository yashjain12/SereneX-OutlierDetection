I had a fun time at my **data science internship at SereneX**, and one of the most involved processes was to experiment with **anomaly detection**, which would be a handy component in our final project: a **Weekly Health Tracker**. For the weekly health tracker, it was of utmost importance to accurately determine whether there would be marginal or significant changes in biometric data of a particular user.

I scraped data using **FitBit API** over a 5 month period for experimentation (Heart Rate, Heart Rate Variability, Step Count, and Quality of Sleep). Then, the aim is to generate more values using different generation techniques, assuming that values are from certain distributions which are unknown. While the **Bootstrapping** Sampling technique can be used for finding these unknown distributions, we aim to find outliers if data points are generated according to two distributions: normal, and random (uniform).

For the first process, we can use **multivariate Gaussian Distributions** to determine the Z-scores at which there is a predicted outlier (a simple example is shown below with one input). However, if the data comes randomly uniformly from a certain interval, we would have to change our method to generate outliers. In such a case, the **[Box-Muller transformation](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)** can be applied, in which a uniform random variable is converted into a normal random variable, which we can find the z-score of.

<p align = "center"><img src = "https://github.com/yashjain12/SereneX-OutlierDetection/assets/20261791/acad545a-39d8-4397-9914-7dada2de1e6a" width = "250"/></p>
<p align = "center">20 percent of data values have |z-score| > 1.28 and can be classified as outliers</p>

Let's look at three of the input features and the actual outliers:
<p align = "center"><img src = "https://github.com/yashjain12/SereneX-OutlierDetection/assets/20261791/e9762bda-fe81-4190-b894-21092225f55a" width = "400"/></p>

First, we experimented with the OpenAI **GPT-4** model, which combined with **LangChain** framework, can feed a dataframe in csv format as input and allow the GPT-4 model to come up with a simple multivariate linear model that minimizes least squared error. Then, we performed **hyperparameter tuning using gridSearchCV** and a scoring function of the form for **Isolation Forest**: <img src = "https://github.com/yashjain12/SereneX-OutlierDetection/assets/20261791/429d3d0e-80b6-4ef7-b8fd-93ac51e82456" width = "150"/>

(where $x$ = data point and $m$ = total number of points, $h(x)$ = search depth for $x$ from the isolate tree, $c(m)$ = Expected value of search depth of a node in a isolate tree)

Linear multivariate regression and isolation forest models were used to find outliers in two datasets: the first dataset had a distribution of step count and heart rate that came from a normal distribution, whereas the second dataset had a distribution of step count and heart rate that came from a random uniform distribution over an interval. Here is the accuracy chart for each model:
<p align = "center"><img src = "https://github.com/yashjain12/SereneX-OutlierDetection/assets/20261791/311ce44d-094c-4469-adcc-7e5747f9215f" width = "400"/></p>

One of the most interesting challenges was overcoming the lack of heart rate variability data. One of the ways I addressed this was **interpolating** the heart rate data (which was given at 1s intervals) to 1ms by using bootstrapping and the **RMSSD** formula to determine heart rate variability over a 5 minute timespan. Following are the results visualized, where blue represents actual HRV, red represents our approximation of HRV:
<p align = "center"><img src = "https://github.com/yashjain12/SereneX-OutlierDetection/assets/20261791/75127ab4-7bf2-4a60-a548-a0745b32593f" width = "300"/></p>

**Let's get to business** and bring out all sorts of (multivariate) functions to tackle this data and find outliers, from exponential tangents to sigmoids to polynomials. The function with the most $r^2$ score can be said to hold the best correlation with the data. Once again, these functions are generally defined, and their coefficients are optimized using the `curve_fit` module.
<p align = "center"><img src = "https://github.com/yashjain12/SereneX-OutlierDetection/assets/20261791/05c5b18c-c1f3-40ed-a309-b1bbbef1ed6a" width = "600"/></p>

Plotting the **ROC-Curves** and computing the Area under the curve (AUC) for accurate implementations of outlier detection:
<p align = "center"><img src = "https://github.com/yashjain12/SereneX-OutlierDetection/assets/20261791/d45caf3d-d4f4-4fd6-8649-59d764cecfd1" width = "500"/></p>

Lastly, let's see the prediction values that the best fitting function (fourth order multivariate polynomial). One can observe that these trend and have a good correlation with the actual outliers:
<p align = "center"><img src = "https://github.com/yashjain12/SereneX-OutlierDetection/assets/20261791/c24b9d0a-1375-4a2b-ba84-af96cf795310" width = "400"/></p>
