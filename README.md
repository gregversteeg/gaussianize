# Gaussianize data


The idea is to apply a smooth, invertible transformation to some univariate data so that the distribution of the
 transformed data is as Gaussian as possible. This would be a pre-processing step for some other analysis. Why might
 this be useful?
 
 1. A standard pre-processing step is to "whiten" data by subtracting the mean and scaling it to have standard deviation 1. 
 Gaussianized data has these properties and more.
 2. Robust statistics / reduce effect of outliers. Lots of real world data exhibits long tails. For machine learning, the small
 number of examples in the tails of the distribution can have a large effect on results. 
 Gaussianized data will "squeeze" the tails in towards the center. 
 3. Theoretical benefits. Gaussian distributions are very well studied with many unique properties that you may like to leverage. 
 4. Various information theoretic quantities are invariant under invertible transforms, so it makes sense to first 
 transform into a friendlier distribution. 
 
 
# Three ways to Gaussianize

## The Lambert Way to Gaussianize heavy-tailed data

For heavy-tailed data, there is a one-parameter family that based on
 Lambert's W function that does this very well. 
This approach is described in Sec. C of the following paper. 
> [The Lambert Way to Gaussianize heavy tailed data with the inverse of Tukey's h transformation as a special case](http://arxiv.org/abs/1010.2265)
> by Georg M. Goerg

[This is already implemented by the author in an R package.](https://cran.r-project.org/web/packages/LambertW/)

The contribution here is to make a (basic) python version that works in the sklearn style. This is very preliminary 
and poorly tested. 

Here's an example of a [QQ plot](https://en.wikipedia.org/wiki/Q–Q_plot) comparing data generated 
from a Cauchy distribution to a normal distribution. 

![Cauchy before](https://github.com/gregversteeg/gaussianize/blob/master/tests/cauchy_before.png?raw=true "Cauchy before")

After applying the transformation, this plot looks like this. 

![Cauchy after](https://github.com/gregversteeg/gaussianize/blob/master/tests/cauchy_after.png?raw=true "Cauchy after")

## Box Cox transformation
This is an old and well known way to [gaussianize data](https://en.wikipedia.org/wiki/Power_transform). 
Some of the limitations of this approach are discussed in Georg's paper, above.  E.g., it only applies to positive data
and gaussianizes data with a heavy right-hand tail. 

This transformation is not yet tested.

## Brute force 

The brute force method I first saw in [this paper](http://www.uv.es/~gcamps/papers/Laparra11.pdf), but I assume it
has been described elsewhere before. The idea is to first make the distribution uniform by rank transforming the data. 
Then you apply the inverse of the CDF for a normal. This results in very gaussian looking empirical data, but it has some
drawbacks. 

One drawback of this approach is that the first step, the empirical copula transform, is not a smooth transformation. 
Furthermore, inverting this transformation can only be done in a kind of piecemeal approximate way which is not yet 
implemented. 

# Example usage

```python
# This example is in tests. 
import gaussianize as g
import numpy as np

x = np.hstack([np.random.standard_cauchy(size=(1000, 2)), np.random.normal(size=(1000, 2))])
out = g.Gaussianize()
out.fit(x)  # Learn the parameters for the transformation
y = out.transform(x)  # Transform x to y, where y should be normal
x_prime = out.invert(y)  # Inverting this transform should recover the data
assert np.allclose(x_prime, x)
out.qqplot(x)  # Plot qq plots for each variable, before and after. 

```

# Command line interface

Preprocess a data file by Gaussianizing each column. The -q option optionally generates qq plots. Default delimiter is 
comma. The default assumption is that the first row and column are labels. This can be altered with options. 

```
python gaussianize.py file.csv --delimiter=',' -o output.csv -q
python gaussianize.py -h  # For a list of all options
python gaussianize.py file.csv --delimiter=',' -o output_brute.csv --strategy='brute' -q
```