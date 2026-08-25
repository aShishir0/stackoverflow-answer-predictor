SELECT
    -- Question fields
    q.id AS question_id,
    q.creation_date,
    q.title,
    q.body,
    q.tags,
    q.owner_user_id,
    q.score AS question_score,
    q.view_count,
    q.answer_count,
    q.comment_count,
    q.accepted_answer_id,

    -- Accepted answer fields
    a.creation_date AS accepted_answer_date,
    a.score AS accepted_answer_score,
    TIMESTAMP_DIFF(a.creation_date, q.creation_date, MINUTE) AS minutes_to_accepted_answer,

    -- First answer fields
    fa.first_answer_date,
    TIMESTAMP_DIFF(fa.first_answer_date, q.creation_date, MINUTE) AS minutes_to_first_answer,

    -- Asker reputation/experience fields
    u.reputation AS asker_reputation,
    u.creation_date AS asker_account_creation_date,
    u.up_votes AS asker_upvote_count,
    u.down_votes AS asker_downvote_count

FROM `bigquery-public-data.stackoverflow.posts_questions` AS q

LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
ON q.accepted_answer_id = a.id

LEFT JOIN (
    SELECT
        parent_id,
        MIN(creation_date) AS first_answer_date
    FROM `bigquery-public-data.stackoverflow.posts_answers`
    GROUP BY parent_id
) AS fa
ON q.id = fa.parent_id

LEFT JOIN `bigquery-public-data.stackoverflow.users` AS u
ON q.owner_user_id = u.id

WHERE q.creation_date BETWEEN TIMESTAMP('2020-09-25') AND TIMESTAMP('2022-09-25')