# Databricks notebook source
# MAGIC %md
# MAGIC # Stage: data collection and importing
# MAGIC ## Import the Linkedin postings 2023-24 from Hugging face.
# MAGIC
# MAGIC Data is collected by the creator using web-scraping.
# MAGIC
# MAGIC This notebook imports the dataset to databricks, saves it as delta data in spark, which is then able to perform SQL queries.

# COMMAND ----------

# install the package which allow to connect to hugging face
%pip install datasets

# COMMAND ----------

# import all the packages
from datasets import load_dataset
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns



# COMMAND ----------

# Load the dataset by its Hugging Face identifier
df1 = load_dataset("lof223/linkedinjobs")
df2 = load_dataset("lof223/job_description")

# COMMAND ----------

# Convert the 'train' split to a Pandas DataFrame
df1_pandas = df1['train'].to_pandas()
df2_pandas = df2['train'].to_pandas()


# Convert Pandas DataFrame to Spark DataFrame
df1_spark = spark.createDataFrame(df1_pandas)
#df2_spark = spark.createDataFrame(df2_pandas) #my server is limited to load this so do the data cleaning first before saving to my catalog

# Save df1 directly
df1_spark.write.format("delta").saveAsTable(
    "my_catalog.linkedin_postings_2023_25.2023_24dtable"
)



# COMMAND ----------

# MAGIC %md
# MAGIC Extract the year data and compare the years covered in both dataset.
# MAGIC df1 covers 2023-24, df2 covers 2021-2023
# MAGIC

# COMMAND ----------


df2_pandas.head(3)
df2_pandas["year"] = df2_pandas["Job Posting Date"].str[:4]
print(df2_pandas[["Job Posting Date", "year"]].head())

# check the number of years in df2
year_count = df2_pandas.groupby(df2_pandas["year"]).size()
print(year_count)
# only 2023 is covered in both, while df1 is the main dataset, for df2 we only want to keep the rows with year 2023.


# COMMAND ----------

# MAGIC %md
# MAGIC Now we want to select only the relevant columns to save the data in the catalog.

# COMMAND ----------

df2_pandas.head(1)

# COMMAND ----------

# the dataset is too large for my server to be save it in the catalogue
# keep only the relevant columns
df2_sub = df2_pandas[
    [
        "Job Title",
        "Company Size",
        'year',
        'Experience'
        ] ]


# Rename column names to be able to save in spark
df2_sub.columns = [
    c.replace(" ", "_").replace("-", "_") for c in df2_sub.columns
]

df2_sub.head(5)

# COMMAND ----------

# Convert to Spark DataFrame
df2_spark = spark.createDataFrame(df2_sub)

# Save as a Delta table in your catalog
df2_spark.write.format("delta").saveAsTable(
    "my_catalog.linkedin_postings_2023_25.df_company_size"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage: Select the relevant columns from the orginal dataset from SQL, and save the sub-datasets.
# MAGIC
# MAGIC I have used SQL to identify the relevant variables, cleaned the job titles and date.
# MAGIC
# MAGIC The next time is to import the sub-data into Python for further data wrangling.
# MAGIC While doing data wrangling in SQL, I have noticed that for 2023-24 dataset, the subset only covers the year 2024.
# MAGIC For 2025, the whole is much smaller than the 2023-24 due to extracting from only LinkedIn for two months.

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage: Clean the sub-data in Python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find the most relevant skills from job desc.
# MAGIC
# MAGIC This section cleans the keywords of relevant skills in the job description. Then the counts of the keywords can be visualised in word cloud in Dashboard.

# COMMAND ----------

# clean the skills data (2025 only)
# import the data

df = spark.read.table('my_catalog.linkedin_postings_2023_25.posting_desc_24')
#display(df)

df_filter = df.filter(df.category != "Other")
display(df_filter)

row_count = df_filter.count()
print(row_count)

# transform into pandas dataframe
df_desc = df_filter.toPandas()


# COMMAND ----------

# check whether df_desc is a pandas dataframe
is_pandas = isinstance(df_desc, pd.DataFrame)
print(is_pandas)

# COMMAND ----------

# filter the rows to only data-realted jobs:
# If you want to filter by index names/values
row_names = ['Data Analyst', 'BI Analyst', ' Data Architect', 'Data Engineer', 'Data Scientis', 'Machine Learning Engineer',
             'Data Administrator','Database Administrator', 'Cybersecurity Analyst']  # your list of row names
df_dt = df_desc[df_desc['category'].isin(row_names)]

df_dt.shape


# COMMAND ----------

# Extract the keywords from the desc col
# Software: SQL, Python, Cloud Platforms

# Define the pattern to match your specified technologies
software = r'(SQL|Python|AWS|Azure|GCP|Cloud Platforms|Git|Javascript)'
databases = r'(SQLite|MySQL|Oracle|PostgreSQL|MongoDB|Redis|Microsoft SQL|MariaDB|Dynamodb)'
soft_skills = r'(problem-solving|communication|analytical thinking|adaptability)'

# Extract all matches and join them into a single string
df_dt['Software'] = df_dt['description'].str.extractall(software).groupby(level=0)[0].apply(lambda x: ', '.join(x))
df_dt['Databases'] = df_dt['description'].str.extractall(databases).groupby(level=0)[0].apply(lambda x: ', '.join(x))
df_dt['Soft_Skills'] = df_dt['description'].str.extractall(soft_skills).groupby(level=0)[0].apply(lambda x: ', '.join(x))


# COMMAND ----------

# count the number of non-missing rows
print(df_dt['Software'].notna().sum())
print(df_dt['Databases'].notna().sum())
print(df_dt['Soft_Skills'].notna().sum())

# COMMAND ----------

# drop the description column
df_dt_clean = df_dt.drop(columns=['description'])

df_dt_clean.head(10)

# COMMAND ----------

# check the year
df_year = df_dt_clean['listed_year'].value_counts().reset_index()
df_year.columns = ['year', 'count']
df_year.head()

# COMMAND ----------

# count the occurence of each keyword
def keyword_counts(df, col_name):
    return (
        df[col_name]
        .dropna()
        .str.split(',\s*')
        .explode()
        .value_counts()
        .reset_index()
        .rename(columns={'index': col_name, col_name: 'count'})
    )

software_counts = keyword_counts(df_dt_clean, 'Software')
database_counts = keyword_counts(df_dt_clean, 'Databases')
soft_skill_counts = keyword_counts(df_dt_clean, 'Soft_Skills')


# COMMAND ----------

# save the counts
def save_to_catalog(df, table_name, catalog="my_catalog.linkedin_postings_2023_25"):
    spark_df = spark.createDataFrame(df)
    spark_df.write.mode("overwrite").saveAsTable(f"{catalog}.{table_name}")

# Usage:
save_to_catalog(software_counts, "software_counts")
save_to_catalog(database_counts, "database_counts")
save_to_catalog(soft_skill_counts, "soft_skill_counts")


# COMMAND ----------

# MAGIC %md
# MAGIC # Stage: Visualisation
# MAGIC ## A stacked bar chart showing the main types of jobs posted in four categories: data-related, directed disrputed, and augmentaton, skill shift and other.
# MAGIC
# MAGIC Firstly, import and clean the data.

# COMMAND ----------

# import 2024 and 2025 datasets and then merge

df_2024 = spark.read.table('my_catalog.linkedin_postings_2023_25.categorised_postings_23_24')
df_2025 = spark.read.table('my_catalog.linkedin_postings_2023_25.categories_25')

# drop irrevalent column
df_2024 = df_2024.drop('avg_cate_salary')
df_2024 = df_2024[df_2024.posting_category != 'Other']
df_2025 = df_2025[df_2025.posting_category != 'Other']

display(df_2024)
display(df_2025)
# Merge the two DataFrames
df_merged = df_2024.union(df_2025)

display(df_merged)

# save the merged dataset for dashboard
df_merged.write.mode('overwrite').saveAsTable(
    'my_catalog.da_profile_dt_dashboards.num_postings_24_25'
)


# COMMAND ----------

# MAGIC %md
# MAGIC Now draw the stacked bar chart.

# COMMAND ----------

# change the df_merged to pandas df
df_merged = df_merged.toPandas()

# Pivot the data to have 'Product' as columns and 'Segment' as the index
pivot_df = df_merged.pivot(index='listed_year',
                    columns='posting_category',
                    values='num_postings')

# Create a grouped barplot
pivot_df.plot.bar(stacked=True,
                  grid=True)

# Add a legend
plt.legend(bbox_to_anchor=(1.04, 1), # shift the legend 4% on the right
           loc='upper left')

plt.xlabel('Number of Postings')
plt.ylabel('Year')
plt.title('Number of Postings by Category and Year')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Stage: Visualisation
# MAGIC ## Pie charts showing the number of postings within data-related jobs and disruptive jobs.

# COMMAND ----------

# import the 2024 data
# for each year, count the number 
df_2024_sub = spark.read.table('my_catalog.linkedin_postings_2023_25.sub_categorised_postings_23_24')
df_2025_sub = spark.read.table('my_catalog.linkedin_postings_2023_25.subcategories_25')

# merge the
display(df_2024_sub)
display(df_2025_sub)

# change the df to pandas df
df_2024_sub = df_2024_sub.toPandas()
df_2025_sub = df_2025_sub.toPandas()


# COMMAND ----------


df_2024_sub = df_2024_sub[df_2024_sub['category'] != 'Other']

# 2024 - throw the categories into three categories
# select the rows for data-related jobs
list_dt = df_2024_sub['category'].isin(['Data Scientist','Machine Learning Engineer', 
                                                   'Data Engineer', 'Data Architect',
                                                   'Data Analyst', 'BI Analyst',
                                                   'Data Administrator', 'Database Administrator', 'Cybersecurity Analyst'])

list_dis = ~df_2024_sub['category'].isin(['Data Scientist','Machine Learning Engineer', 
                                                   'Data Engineer', 'Data Architect',
                                                   'Data Analyst', 'BI Analyst',
                                                   'Data Administrator', 'Database Administrator', 'Cybersecurity Analyst'])


# data-related jobs - 2024
data_24 = df_2024_sub[list_dt]
print(data_24)

# disruption and augmentation and skill uplift jobs - 2024
dis_24 = df_2024_sub[list_dis]
print(dis_24)

# only data-related jobs - 2025
jobs_25 = df_2025_sub[df_2025_sub['category'] != 'Other']

# count the number of postings by job title
data_25 = jobs_25['category'].value_counts().reset_index()
print(data_25)



# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the datasets for pie chart for dashboard.

# COMMAND ----------

# Define a list of tuples: (pandas_dataframe, target_table_name)
dataframes = [
    (data_24, 'data_24'),
    (dis_24, 'dis_24')
]

# Define the base catalog and schema path
base_table_path = 'my_catalog.da_profile_dt_dashboards.'

# Loop through each DataFrame and save as a Spark table
for pandas_df, table_name in dataframes:
    # Convert pandas DataFrame to Spark DataFrame
    spark_df = spark.createDataFrame(pandas_df)
    
    # Save to the specified table
    spark_df.write.mode('overwrite').saveAsTable(
        f'{base_table_path}{table_name}'
    )
    print(f"Saved {table_name} to Spark table successfully")
    

# COMMAND ----------

print(data_24)

# COMMAND ----------

# Now produce three pie charts
# Create a color dictionary
custom_colors = {
    'Data Analyst': '#1f77b4',   # Blue
    'Data Scientist': '#ff7f0e', # Orange
    'Data Engineer': '#2ca02c',  # Green
    'Machine Learning Engineer': '#d62728', # Red
    'BI Analyst': '#9467bd'      # Purple
}

# Get default matplotlib colors (one for each category)
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Create a final color list: use custom colors where defined, else defaults
final_colors = []
for i, cat in enumerate(data_24['category']):
    # Use custom color if available; else use the i-th default color
    final_colors.append(custom_colors.get(cat, default_colors[i % len(default_colors)]))

plt.figure(figsize=(12,12))
plt.pie(
   data_24['num_postings'], labels = data_24['category'],
    colors=final_colors, 
   autopct='%1.1f%%'
)

plt.title('Distribution of Data-related jobs (2024)')
plt.show()


# Extract colors in the order of `categories`
colors = [custom_colors[cat] for cat in data_25['index']]

plt.figure(figsize=(12,12))
plt.pie(
   data_25['category'], labels = data_25['index'], 
   colors=colors,
   autopct='%1.1f%%'
)

plt.title('Distribution of Data-related jobs (2025)')
plt.show()



plt.figure(figsize=(13,13))
plt.pie(
   dis_24['num_postings'], labels = dis_24['category'], 
   autopct='%1.1f%%'
)

plt.title('Distribution of Disrupted or Augmentation and Skill Shift jobs (2024)')
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC # Stage Visualisation
# MAGIC ### A word cloud chart for skills required in data-related jobs.

# COMMAND ----------

# install the package
%pip install wordcloud

# import the package
from wordcloud import WordCloud


# COMMAND ----------

# import the cleaned counts data
software_counts = spark.read.table("my_catalog.linkedin_postings_2023_25.software_counts")
softskill_counts = spark.read.table("my_catalog.linkedin_postings_2023_25.soft_skill_counts")
database_counts = spark.read.table ("my_catalog.linkedin_postings_2023_25.database_counts")

# covert each to pandas dataframe
software_counts = software_counts.toPandas()
softskill_counts = softskill_counts.toPandas()
database_counts = database_counts.toPandas()

#display
print(software_counts)
print(softskill_counts)
print(database_counts)


# COMMAND ----------

# lower max_font_size, change the maximum number of word and lighten the background:
# List of tuples: (DataFrame, chart_title)
dataframes = [
    (software_counts, 'Mostly required Sofware Skills','Software'),
    (softskill_counts, 'Mostly required Soft Skills', 'Soft_Skills'),
    (database_counts, 'Mostly used Databases', 'Databases')
]

# Loop through each DataFrame in the list
for df, title, word_col in dataframes:
    # Create frequency dictionary from the DataFrame
    word_freq = df.set_index(word_col)['count'].to_dict()
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white'
    ).generate_from_frequencies(word_freq)
    
    # Plot and display
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=14)
    plt.axis('off')  # Hide axes
    plt.show()  # Display the plot
    

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage:Visualisation
# MAGIC ### A bar chart showing the average annual salary for each data-related job category.
# MAGIC

# COMMAND ----------

# Load the table
df = spark.read.table("my_catalog.linkedin_postings_2023_25.data_postings_23_views_applies")

# Convert to pandas DataFrame
df_pd = df.toPandas()


# COMMAND ----------

# Categorise data-related jobs into broader categories
df_pd = df_pd[df_pd['category'] != 'Other']

# Define exact job title to category mapping
job_category_map = {
    # Data Infrastructure
    'Data Engineer': 'Data Infrastructure',
    'Data Architect': 'Data Infrastructure',
    
    # Data Analysis
    'Data Analyst': 'Data Analysis', 
    'BI Analyst': 'Data Analysis',
    
    # Data Science and AI
    'Data Scientist': 'Data Science and AI',
    'Machine Learning Engineer': 'Data Science and AI',
    
    # Data Management and Security
    'Data Administrator': 'Data Management and Security',
    'Database Administrator': 'Data Management and Security',
    'Cybersecurity Analyst': 'Data Management and Security'
}

# Apply the mapping
df_pd['broad_category'] = df_pd['category'].map(job_category_map)

# Fill any unmatched jobs with 'Other'
df_pd['broad_category'] = df_pd['broad_category'].fillna(df_pd['category'])

df_pd.head(10)

# COMMAND ----------

# change all the salaries to annual
df_pd['avg_salary'] = df_pd['avg_salary'] * df_pd['pay_period'].map({'YEARLY': 1, 'MONTHLY': 12, 'WEEKLY': 52, 'HOURLY': 2080})

# keep only entry-level and associate level salaries
df_pd_entry = df_pd[df_pd['formatted_experience_level'].isin(['Entry level', 'Associate'])] 
df_pd_entry.describe()

# COMMAND ----------

# keep mid-senior level
df_pd_mid = df_pd[df_pd['formatted_experience_level'] == 'Mid-Senior level']
# drop outliers
df_pd_mid = df_pd[df_pd['avg_salary'] < 1000000]
df_pd_mid.describe()

# COMMAND ----------

# find the mean of entry-level salaries
mean_df_entry = df_pd_entry.groupby('broad_category')['avg_salary'].agg(['mean','count']).reset_index()
mean_df_entry.columns = ['broad_category', 'mean_salary','count']
print(mean_df_entry)

# find the mean mid-senior level salaries
mean_df_mid = df_pd_mid.groupby('broad_category')['avg_salary'].agg(['mean','count']).reset_index()
mean_df_mid.columns = ['broad_category', 'mean_salary', 'count']
print(mean_df_mid)

# COMMAND ----------

# for consistency, keep the same postings for salaries in entry and mid-senior level
mean_df_entry = mean_df_entry[mean_df_entry['broad_category'] != 'Lawyer']
mean_df_mid = mean_df_mid[mean_df_mid['broad_category'].isin(['Accounting Clerk','Bookkeeping','Customer Service','Data Analysis', 'Data Infrastructure','Data Management and Security',
                                                            'Data Science and AI','HR', 'Underwriter'])]

# COMMAND ----------

for mean_df, level in zip([mean_df_entry, mean_df_mid], ['Entry/Associate', 'Mid-Senior']):
    # sort the order of the bars
    mean_df_sorted = mean_df[mean_df['mean_salary'] > 0].sort_values('mean_salary', ascending=True)
    
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=mean_df_sorted,
        x='mean_salary',
        y='broad_category',
        palette='viridis'
    )
    plt.title(f"Average Annual Salary ($) by Broad Category - {level} Level")
    plt.xlabel("Mean Salary")
    plt.ylabel("Broad Category")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the salary data for dashboard.

# COMMAND ----------

for df, name in zip([mean_df_entry, mean_df_mid], ['mean_df_entry', 'mean_df_mid']):
    spark_df = spark.createDataFrame(df)
    spark_df.write.mode('overwrite').saveAsTable(
        f'my_catalog.da_profile_dt_dashboards.{name}'
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage: Visualisation
# MAGIC ## Correlation between views or applies and salary for each experience level
# MAGIC

# COMMAND ----------

# drp avg_salary below 50000 as these are not monthly salaries
df_pd = df_pd[df_pd["avg_salary"] >= 30000]

counts =  df_pd.groupby("pay_period").size()
print(counts)

# COMMAND ----------

# Plot scatter plot
plt.figure(figsize=(10,8))
sns.scatterplot(
    data=df_pd,
    x="views",
    y="avg_salary",
    hue = 'formatted_experience_level'
   
)
plt.title("Annual Salary ($) vs Views by Experience Level")
plt.xlabel("Views")
plt.ylabel("Annual Salary ($)")
plt.legend(title="Experience Level")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
sns.scatterplot(
    data=df_pd,
    x="applies",
    y="avg_salary",
    hue = 'formatted_experience_level'
   
)
plt.title("Annual Salary ($) vs Applies by Experience Level")
plt.xlabel("Applies")
plt.ylabel("Annual Salary ($)")
plt.legend(title="Experience Level")
plt.tight_layout()
plt.show()


# COMMAND ----------

