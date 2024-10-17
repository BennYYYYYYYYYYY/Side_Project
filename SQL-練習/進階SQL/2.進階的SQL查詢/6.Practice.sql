/*
在本機 MySQL Server 的 covid19 資料庫，
建立一個檢視表 date_difference 顯示 accumulative_cases 資料表與 calendars 資料表兩者的日期資料差集，
亦即哪些日期在 calendars 資料表之中，卻沒有出現在 accumulative_cases 資料表中。
 */
CREATE VIEW covid19.date_difference
as
SELECT id,
	   recorded_on
FROM covid19.calendars
WHERE id IN (
			SELECT id
			FROM covid19.calendars
			EXCEPT
			SELECT DISTINCT calendar_id
			FROM covid19.accumulative_cases);

SELECT * 
FROM covid19.date_difference;


/*
在本機 MySQL Server 的 tw_election_2020 資料庫，
建立一個檢視表 presidential_votes_percentage 顯示每組總統候選人的得票比例。
 */
CREATE VIEW tw_election_2020.presidential_votes_percentage
AS 
WITH votes_by_candidate AS (
  SELECT candidates.num,
         candidates.name,
         SUM(presidential.votes) AS votes
  FROM tw_election_2020.presidential
  JOIN tw_election_2020.candidates
  ON presidential.candidate_id = candidates.id  -- 修正拼寫錯誤
  GROUP BY candidates.id
)
SELECT *,
       SUM(votes) OVER () AS total_votes,
       votes / (SUM(votes) OVER ()) AS votes_percentage
FROM votes_by_candidate;


/*
在本機 MySQL Server 的 tw_election_2020 資料庫，
建立一個檢視表 legislative_at_large_votes_percentage 顯示每個政黨的不分區立委得票比例。
 */
CREATE VIEW tw_election_2020.legislative_at_large_votes_percentage
AS
WITH votes_by_party AS (
  SELECT parties.name,
         SUM(legislative_at_large.votes) AS votes
  FROM tw_election_2020.legislative_at_large
  JOIN tw_election_2020.parties
  ON legislative_at_large.party_id = parties.id
  GROUP BY parties.id
  ORDER BY votes DESC
)
SELECT *,
       SUM(votes) OVER () AS total_votes,
       votes / (SUM(votes) OVER ()) AS vote_percentage  -- 修正拼寫和符號
FROM votes_by_party;

SELECT *
FROM tw_election_2020.legislative_at_large_votes_percentage;



/*
在本機 MySQL Server 的 covid19 資料庫，
建立一個檢視表 most_populated_province 顯示 locations 資料表每個國家人口數最多的州別（或省份），
沒有切分州別（或省份）資料的國家不需要包含在該檢視表中。
 */
CREATE VIEW covid19.most_populated_province
AS
SELECT DISTINCT country_name,
FIRST_VALUE(province_name) OVER (PARTITION BY country_name ORDER BY population DESC) AS most_populated_province_name
FROM covid19.locations
WHERE province_name IS NOT NULL
ORDER BY country_name;

SELECT * 
FROM covid19.most_populated_province



/*
在本機 MySQL Server 的 covid19 資料庫，
建立一個檢視表 daily_cases 顯示每個 location_id、calendar_id 的每日新增個案數。
 */
CREATE VIEW covid19.daily_cases
AS
SELECT location_id,
       calendar_id,
       confirmed - LAG(confirmed) OVER (PARTITION BY location_id ORDER BY calendar_id) AS confirmed,  -- 修正拼寫錯誤
       deaths - LAG(deaths) OVER (PARTITION BY location_id ORDER BY calendar_id) AS deaths  -- 統一列名
FROM covid19.accumulative_cases;

SELECT count(*) 
FROM covid19.daily_cases;















 
 
 