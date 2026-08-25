SELECT
  MIN(creation_date) AS earliest_question_date,
  MAX(creation_date) AS latest_question_date
FROM `bigquery-public-data.stackoverflow.posts_questions`