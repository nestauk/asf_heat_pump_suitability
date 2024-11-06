# %%
# Check on R install - made to my heat pump suitability environment.
import rpy2.situation
for row in rpy2.situation.iter_info():
    print(row)

# %% [markdown]
# ### Adding mlfit to R environment
# 
# To install `mlfit` I had to navigate to the R installation listed above, launch R and run:
# 
# ```{R}
# install.packages("remotes", repos="https://www.stats.bris.ac.uk/R/")
# library(remotes)
# install_version("Matrix", version="1.6-5", repos="https://www.stats.bris.ac.uk/R/")
# install.packages("mlfit", repos="https://www.stats.bris.ac.uk/R/")
# ```
# This was necessary as the version of R installed wasn't recent enough for the latest version of Matrix.

# %%
# Check the mfit package is available
import rpy2.robjects.packages as rpackages

# It it appears in thsi list, all is well!
list(filter(lambda x: 'mlfit' in x[0], rpackages.InstalledPackages()))

# %% [markdown]
# ### mlfit relevant example
# 
# In this code, we'll execute the relevant mlfit example but in python code, passing the data and relying on R only for the mlfit call.

# %%
import pandas
import geopandas
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

# %%
# Example mlfit data as pandas dataframes.
reference_sample = pandas.DataFrame(data = {'HHNR': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5,
                                                     5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8],
                                            'PNR': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                                            'APER': [3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                                                     3, 3, 2, 2, 5, 5, 5, 5, 5, 2, 2],
                                            'CAR': ["0", "0", "0", "0", "0", "0", "0", "0", "1",
                                                    "1", "1", "1", "1", "1", "1", "1", "1", "1",
                                                    "1", "1", "1", "1", "1", ],
                                            'WKSTAT': ["1", "2", "3", "1", "3", "1", "1", "2",
                                                       "1", "3", "3", "2", "2", "3", "1", "2",
                                                       "1", "1", "2", "3", "3", "1", "2"]})

individual_control = pandas.DataFrame(data = {'WKSTAT': ["1", "2", "3"], 'N': [91, 65, 104]})

group_control = pandas.DataFrame(data = {'CAR': ["0", "1"], 'N': [35, 65]})

# %%
# convert pandas dataframes to R dataframes.
with (robjects.default_converter + pandas2ri.converter).context():
  r_reference_sample = robjects.conversion.get_conversion().py2rpy(reference_sample)
  r_individual_control = robjects.conversion.get_conversion().py2rpy(individual_control)
  r_group_control = robjects.conversion.get_conversion().py2rpy(group_control)

# %%
# register r dataframes within r isntance global environment
robjects.globalenv['reference_sample'] = r_reference_sample
robjects.globalenv['individual_control'] = r_individual_control
robjects.globalenv['group_control'] = r_group_control

# %%
# create mlfit fitting problem and solve in r.
fit = robjects.r(
    """
library(mlfit)

fitting_problem <- ml_problem(
  ref_sample = reference_sample, 
  controls = list(
    individual = list(individual_control),
    group = list(group_control)
  ),
  field_names = special_field_names(
    groupId = "HHNR", 
    individualId = "PNR", 
    count = "N"
  )
)

ml_fit(ml_problem = fitting_problem, algorithm = "hipf")
"""
)

# %%
# Get weights from r fit object as python list.
weights = [val for val in fit.rx2('weights')]

# %%
# Check the calculated weights work against the reference sample 
reference_sample['weights'] = weights

# %%
individual_control

# %%
# Individual weights sum to individual controls
reference_sample.groupby('WKSTAT')['weights'].sum()

# %%
group_control

# %%
# mean household weights sum to group controls
reference_sample.groupby('HHNR', as_index=False).agg({'weights':'mean', 'CAR':'first'}).groupby('CAR')['weights'].sum()

# %% [markdown]
# ### Extending the example to the heat pump suitability context
# 
# The difference between the above problem and our problem is that the above problem is fitting data where characteristics would logically be observed/measured at the different levels, whereas in our problem data should all logically be observed at the individual-level, but some of it is only available at a more aggregate level. This means we'll have to scale the margins at the higher level according to the nesting at the lower level.
# 
# Unfortunately, this doesn't work as the nesting isn't sufficiently rich enough - you just can't represent the la level as you can't have repeated lsoa constraints.

