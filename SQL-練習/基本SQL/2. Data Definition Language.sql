# 建立 cloned_covid19 資料庫，charset 指定 utf8mb4，collation 指定 utf8mb4_unicode_ci
CREATE DATABASE cloned_covid19 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
SHOW DATABASES;


# 建立新資料表
CREATE TABLE cloned_imdb.movies(
	id int UNSIGNED,
	title varchar(200),
	rating float,
	release_year YEAR,		# year: yyyy
	release_date date,		# date: 'YYYY-MM-DD'
	release_time time		# time: 'hh:mm:ss'
    );


# 在 cloned_covid19 資料庫中，建立 accumulative_cases 資料表
CREATE TABLE cloned_covid19.accumulative_cases(
	id int unsigned,
	calendar_id int unsigned,
	location_id int unsigned,
	confirmed bigint,
	deaths bigint
);


# 在已經存在的資料表中新增一個名為 title 的欄位
ALTER TABLE cloned_imdb.movies ADD title varchar(200);


# 從資料表中刪除名為 title 的欄位
ALTER TABLE cloned_imdb.movies DROP title;


# 將資料表中的 id 欄位重新命名為 movie_id
ALTER TABLE cloned_imdb.movies RENAME COLUMN id TO movie_id;


# 將資料表中的 movie_id 欄位修改為 tinyint unsigned 類型
ALTER TABLE cloned_imdb.movies MODIFY COLUMN movie_id tinyint unsigned;


# 清空資料表中的所有數據
TRUNCATE TABLE cloned_imdb.movies;


# 在 covid19 資料庫中，篩選 accumulative_cases.calendar_id = 1164 建立 accumulative_cases_20230309 檢視表
CREATE VIEW covid19.accumulative_cases_20230309
 AS 
SELECT *
 FROM covid19.accumulative_cases
 WHERE calendar_id = 1164;


# 在 covid19 資料庫中，篩選 locations.iso3 IN ('TWN', 'JPN') 建立 locations_twn_jpn 檢視表
CREATE VIEW covid19.locations_twn_jpn
AS
SELECT *
FROM covid19.locations
WHERE iso3 IN ('TWN', 'JPN');


# 在 covid19 資料庫中，建立 views_joined 檢視表，使用 JOIN 連接兩個檢視表。
# 以 accumulative_cases_20230309 作為左檢視表、locations_twn_jpn 作為右檢視表
CREATE VIEW covid19.views_joined
AS
SELECT locations_twn_jpn.country_name,
	   locations_twn_jpn.province_name,
	   accumulative_cases_20230309.confirmed,
	   accumulative_cases_20230309.deaths
FROM covid19.accumulative_cases_20230309
JOIN covid19.locations_twn_jpn
ON accumulative_cases_20230309.location_id = locations_twn_jpn.id


# 在 covid19 資料庫中，建立 views_left_joined 檢視表，使用 LEFT JOIN 連接兩個檢視表。
# 以 accumulative_cases_20230309 作為左檢視表、locations_twn_jpn 作為右檢視表。
CREATE VIEW covid19.views_left_joined
AS 
SELECT locations_twn_jpn.country_name,
	   locations_twn_jpn.province_name,
	   accumulative_cases_20230309.confirmed,
	   accumulative_cases_20230309.deaths
FROM covid19.accumulative_cases_20230309
LEFT JOIN covid19.locations_twn_jpn
ON accumulative_cases_20230309.location_id = locations_twn_jpn.id


# 在 covid19 資料庫中，建立 views_right_joined 檢視表。
# 使用 RIGHT JOIN 連接兩個檢視表，以 accumulative_cases_20230309 作為左檢視表、locations_twn_jpn 作為右檢視表
CREATE VIEW covid19.views_right_joined
AS 
SELECT locations_twn_jpn.country_name,
	   locations_twn_jpn.province_name,
	   accumulative_cases_20230309.confirmed,
	   accumulative_cases_20230309.deaths
FROM covid19.accumulative_cases_20230309
RIGHT JOIN covid19.locations_twn_jpn
ON accumulative_cases_20230309.location_id = locations_twn_jpn.id



















































































