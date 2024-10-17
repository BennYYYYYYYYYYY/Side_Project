# 建立 cloned_covid19 資料庫，charset 指定 utf8mb4，collation 指定 utf8mb4_unicode_ci
CREATE DATABASE cloned_covid19 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;



# 在資料表中添加新資料
CREATE TABLE cloned_imdb.movies(
	id int unsigned,
	title varchar(200),
	release_year YEAR,
	rating float 
);

INSERT INTO cloned_imdb.movies (id, title, release_year, rating) VALUES
	(1, 'The Shawshank Redemption', 1994, 9.3);



# 在 cloned_covid19 資料庫中，將 locations 資料表中 province_name 欄位原為空字元 '' 的觀測值更新為 NULL
SELECT *
FROM cloned_covid19.locations
WHERE province_name = '';

UPDATE cloned_covid19.locations
SET province_name = NULL
WHERE province_name = 'NULL';

SELECT *
FROM cloned_covid19.locations
WHERE province_name IS NULL;



# 在 cloned_covid19 資料庫中，將 locations 資料表中 iso2 欄位原為空字元 '' 的觀測值更新為 NULL
# 在 cloned_covid19 資料庫中，將 locations 資料表中 iso3 欄位原為空字元 '' 的觀測值更新為 NULL
SELECT *
FROM cloned_covid19.locations
WHERE iso2 = '';

UPDATE cloned_covid19.locations
SET iso2 = NULL,
	iso3 = NULL
WHERE iso2 = '' AND iso3 = '';



# 在資料表中為 id 欄位新增一個唯一性約束 (UNIQUE)，這表示 id 欄位中的每個值必須是唯一的，不能重複。
ALTER TABLE cloned_imdb.movies
 ADD CONSTRAINT UNIQUE (id);



# 修改資料表中的 id 欄位，使其變為無符號整數類型且不允許為空值
ALTER TABLE cloned_imdb.movies
 MODIFY id int UNSIGNED NOT NULL;  



# 修改資料表中的 id 欄位，使其變為自增整數類型
ALTER TABLE cloned_imdb.movies
 MODIFY id int UNSIGNED AUTO_INCREMENT;  


# 在資料表的 release_year 欄位上建立索引，索引名稱為idx_release_year，以加快基於release_year欄位的查詢速度
CREATE INDEX idx_release_year
	ON clone_imdb.movies(release_year);



# 在資料表上刪除名為 idx_release_year 的索引
DROP INDEX idx_release_year
	ON clone_imdb.movies;



# 從資料表中刪除 id 為4的那一行記錄
DELETE FROM clone_imdb.movies
WHERE id = 4;



# 在 cloned_covid19 資料庫中，將 locations 資料表中的 id 欄位增添約束條件 PRIMARY KEY
ALTER TABLE cloned_covid19.locations
ADD CONSTRAINT PRIMARY KEY (id); 

SHOW INDEX FROM cloned_covid19.locations; 



# 在 cloned_covid19 資料庫中，將 accumulative_cases 資料表中的 id 欄位增添約束條件 PRIMARY KEY
ALTER TABLE cloned_covid19.accumulative_cases
ADD CONSTRAINT PRIMARY KEY (id);

SHOW INDEX FROM cloned_covid19.accumulative_cases;



# 在 cloned_covid19 資料庫中，將 accumulative_cases 資料表中的 location_id 欄位增添 fk_accumulative_cases_locations 約束條件 FOREIGN KEY 參照 locations 資料表的 id
ALTER TABLE cloned_covid19.accumulative_cases
ADD CONSTRAINT fk_accumulative_cases_locations FOREIGN KEY (location_id) REFERENCES locations (id); 

SHOW INDEX FROM cloned_covid19.accumulative_cases;
