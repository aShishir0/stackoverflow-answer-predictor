from google.cloud import bigquery

client = bigquery.Client(project="stackoverflow-predictor")

query = """
SELECT COUNT(*) as n
FROM `bigquery-public-data.stackoverflow.posts_questions`
LIMIT 1
"""

result = client.query(query).to_dataframe()
print(result)