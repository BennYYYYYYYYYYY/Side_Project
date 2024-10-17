/*
1. CROSS JOIN
SQL中的一種笛卡爾積（Cartesian Product）操作，它會將兩個資料表的每一行與另一個表的每一行進行配對，產生所有可能的組合。
這通常會返回大量的結果，因為如果表A有m行，表B有n行，結果會有m * n行。

2. SELF JOIN
表自己與自己進行的JOIN操作，用於在同一個表中找到關聯數據。
這通常需要為表設定不同的別名來區分兩個表的數據。
*/


-- CROSS JOIN: 結果會是 (2*3 = 6)，且不需要用ON
SELECT directors.name AS director_name,
	   actors.name AS actor_name
FROM imdb.directors   -- 連結資料表1
CROSS JOIN imdb.actors  -- 連結資料表2
WHERE directors.name IN ('Steven Spielberg', 'Christopher Nolan') AND
	  actors.name IN ('Tom Hanks', 'Tom Cruise', 'Aamir Khan');
	  

-- SELF JOIN應用場景：查詢同名同姓的演員，但可讀性不佳
SELECT DISTINCT a1.name, -- DISTINCT: 篩選查詢結果中的重複記錄，只返回唯一的結果。
				a1.link
FROM imdb.actors AS a1,
	 imdb.actors AS a2
WHERE a1.name = a2.name AND
	  a1.link != a2.link;

-- 改寫 SELF JOIN，將可讀性提升
SELECT name,
	   link
FROM imdb.actors
WHERE name IN (
	SELECT name    -- 把同名同姓但LINK不一樣的人抓出來
	FROM imdb.actors
	GROUP BY name        -- COUNT: 一個聚合函數，用來計算結果集中行的數量。
	HAVING COUNT(*) > 1  -- HAVING: 用來對GROUP BY的分組結果進行篩選，通常與聚合函數（如COUNT()、SUM()等）搭配使用。
)
ORDER BY name;
	 
	 
	 
	 
	 
	 
	 
	 
	 
	 
	 
	 
	 
	 
	 
	 
	 