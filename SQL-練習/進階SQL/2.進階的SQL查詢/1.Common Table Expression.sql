/*
通用資料表運算式（Common Table Expression, CTE)：
	可以將通用資料表運算式視為一種暫存的檢視表，僅在一個 SQL 敘述中有效。
	通用資料表運算式必須在同一個 SQL 敘述中涵蓋定義與查詢的語法。	
*/

WITH avg_rating_by_release_year AS ( # 使用 WITH 建立通用資料表運算式並給予命名
	SELECT release_year, # 加入 SQL 敘述作為通用資料表運算式的內容
		   avg(rating) AS avg_rating
	FROM imdb.movies
	GROUP BY release_year
)
SELECT * # 從通用資料表運算式中查詢資料。
FROM avg_rating_by_release_year 
WHERE avg_rating >= 8.7;



/*
透過通用資料表運算式複習 MySQL 不同的連接類型 (JOIN、LEFT JOIN、RIGHT JOIN)
*/
WITH casting_shawshank_darkknight AS ( # 第一個 CTE
	SELECT *
	FROM imdb.movies_actors
	WHERE movie_id IN (1, 3)
),
movies_shawshank_forrest AS ( # 第二個 CTE
	SELECT *
	FROM imdb.movies
	WHERE title IN ('The Shawshank Redemption', 'Forrest Gump') 
)
SELECT movies_shawshank_forrest.title,  # 建立 JOIN    
       casting_shawshank_darkknight.actor_id  
FROM casting_shawshank_darkknight
JOIN movies_shawshank_forrest   # 可以測試不同的 JOIN(JOIN/LEFT/RIGHT)
ON casting_shawshank_darkknight.movie_id = movies_shawshank_forrest.id;








