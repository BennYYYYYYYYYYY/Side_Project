/*
1. Window Function(視窗函數)
	用來在不改變查詢結果行數的情況下，對查詢的某個範圍（稱為「窗口」）內的數據進行計算。
	與聚合函數不同，Window Function 不會將多行結果合併成一行，而是為每一行生成計算結果，並保留每一行的原始數據。


2. 如何使用 window function
	1. OVER()
	2. PARTITION BY：把資料劃分為不同的「小組」，每個小組內的資料獨立進行計算，保留每一行的資料。
	3. GROUP BY()：控制資料的排序順序。控制的是分組內的數據如何進行排序，從而決定窗口函數的計算順序。
	
	
3. 一般聚合的 Window function
	

4. 具有排序功能的視窗函數
	1. ROW_NUMBER()：相同數值皆給予排序 (很像index) 
	2. RANK()：相同數值給予重複排序 (可並列名次)
	3. FIRST_VALUE()/LAST_VALUE()/NTH_VALUE()：排名第一、最後、第 N 位的數值。
	4. 搭配具有排序功能視窗函數的時候加入 ORDER BY 進行排序。
		
		
5. LEAD() 與 LAG() 函數
	1. LEAD()：獲取後一行的值。 
	2. LAG()：獲取前一行的值。
	
	
	
*/


-- 使用聚合函數情況 (會將多行結果合併為一行)
SELECT rating,
       COUNT(*) AS number_of_movies
FROM imdb.movies
GROUP BY rating;


-- 1. 一般聚合 window function (會顯示多行結果) 
SELECT title,  -- 可以拿來檢查其他項目，與在rating相同的結果 (因為window func會把結果都列出)
	   rating,
	   count(*) OVER (PARTITION BY rating) AS number_of_movies_by_rating
FROM imdb.movies
ORDER BY rating DESC
LIMIT 10;


-- 2. 具有排序功能的 window function： ROW_NUMBER()、RANK()
SELECT title,
	   rating,
	   ROW_NUMBER() OVER (ORDER BY rating DESC) AS row_num, -- index
	   RANK() OVER (ORDER BY rating) AS rating_rank -- 名次(可並列)
FROM imdb.movies
LIMIT 10;


-- 2. 具有排序功能的 window function： FIRST_VALUE()、LAST_VALUE()、NTH_VALUE()
-- PARTITION BY release_year：這是用來將電影按 release_year 進行分組，每一年的電影組成一個「窗口」，讓窗口函數 FIRST_VALUE() 在每一年的電影內部進行操作，而不是在整個資料集上操作。
SELECT title,
       release_year,
       rating,
       FIRST_VALUE(rating) OVER (PARTITION BY release_year ORDER BY rating DESC) AS highest_rating_by_year,
       LAST_VALUE(rating) OVER (PARTITION BY release_year) AS lowest_rating_by_year,
       NTH_VALUE(rating, 2) OVER (PARTITION BY release_year) AS second_highest_rating_by_year
  FROM imdb.movies
 ORDER BY release_year DESC
 LIMIT 10;


-- 3. LAG()、LEAD()應用：LAG() 用來獲取台灣每一天的累計確診和累計死亡數據的前一天的值，便於與當當天的數據進行相減，得到當日新稱的數據。
SELECT calendars.recorded_on,
       locations.country_name AS country,
       accumulative_cases.confirmed AS accumulative_confirmed,
       accumulative_cases.confirmed - LAG(accumulative_cases.confirmed) OVER (PARTITION BY accumulative_cases.location_id ORDER BY calendars.recorded_on) AS daily_confirmed,
       accumulative_cases.deaths AS accumulative_deaths,
       accumulative_cases.deaths - LAG(accumulative_cases.deaths) OVER (PARTITION BY accumulative_cases.location_id ORDER BY calendars.recorded_on) AS daily_deaths
  FROM covid19.accumulative_cases
  JOIN covid19.calendars
    ON accumulative_cases.calendar_id = calendars.id
  JOIN covid19.locations
    ON accumulative_cases.location_id = locations.id
 WHERE locations.country_name = 'Taiwan'
 LIMIT 30;











