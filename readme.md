# The Lambert Way to Gaussianize heavy-tailed data

The idea is to apply a smooth, invertible transformation of some univariate data so that the distribution of the
 transformed data is as Gaussian as possible. For heavy-tailed data, there is a one-parameter family that based on
 Lambert's W function that does this very well. 
This approach is described in Sec. C of the following paper. 
> [The Lambert Way to Gaussianize heavy tailed data with the inverse of Tukey's h transformation as a special case](http://arxiv.org/abs/1010.2265)
> by Georg M. Goerg

[This is already implemented by the author in an R package.](https://cran.r-project.org/web/packages/LambertW/)

The contribution here is to make a (basic) python version that works in the sklearn style. This is very preliminary 
and poorly tested. 

Here's an example of a QQ plot comparing data generated from a Cauchy distribution to a normal distribution. 

![Cauchy before](https://github.com/gregversteeg/gaussianize/blob/master/tests/cauchy_before.png?raw=true "Cauchy before")

After applying the transformation, this plot looks like this. 

![Cauchy after](https://github.com/gregversteeg/gaussianize/blob/master/tests/cauchy_after.png?raw=true "Cauchy after")

## Example usage

```python
# This example is in tests. 
import gaussianize as g
import numpy as np

x = np.hstack([np.random.standard_cauchy(size=(1000, 2)), np.random.normal(size=(1000, 2))])
out = g.Lambert()
out.fit(x)  # Learn the parameters for the transformation
y = out.transform(x)  # Transform x to y, where y should be normal
x_prime = out.invert(y)  # Inverting this transform should recover the data
assert np.allclose(x_prime, x)
out.qqplot(x)  # Plot qq plots for each variable, before and after. 

```