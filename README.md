I had a fun time at my data science internship at SereneX, and one of the most involved processes was to experiment with anomaly detection, which would be a handy component in our final project: a Weekly Health Tracker. For the weekly health tracker, it was of utmost importance to accurately determine whether there would be marginal or significant changes in biometric data of a particular user.

I scraped data using FitBit API over a 5 month period for experimentation (Heart Rate, Heart Rate Variability, Step Count, and Quality of Sleep). Then, the aim is to generate more values using different generation techniques, assuming that values are from certain distributions which are unknown. While the Bootstrapping Sampling technique can be used for finding these unknown distributions, we aim to find outliers if data points are generated according to two distributions: normal, and random (uniform).

For the first process, we can use multivariate Gaussian Distributions to determine the Z-scores at which there is a predicted outlier (a simple example is shown below with one input). However, if the data comes randomly uniformly from a certain interval, we would have to change our method to generate outliers. In such a case, the [Box-Muller](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform) transformation can be applied, in which a uniform random variable is converted into a normal random variable, which we can find the z-score of.

<p align = "center"><img src = "https://github.com/yashjain12/SereneX-OutlierDetection/assets/20261791/acad545a-39d8-4397-9914-7dada2de1e6a" width = "250"/></p>
<p align = "center">20 percent of data values have |z-score| > 1.28 and can be classified as outliers</p>

Let's look at three of the input features and the actual outliers:
<p align = "center"><img src = "https://github.com/yashjain12/SereneX-OutlierDetection/assets/20261791/e9762bda-fe81-4190-b894-21092225f55a" width = "250"/></p>
