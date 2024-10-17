/*
1. Metadata
	在SQL中，Metadata（元資料）就是「關於資料的資料」。它不是資料本身，而是描述資料庫結構的資訊。
	假設，資料是書裡的內容，而元資料是書的目錄，告訴你書的章節、標題、每章的頁數等等。


2. 查詢 Metadata 資料庫
	系統資料庫 "information_schema"，它存儲了所有資料庫、表和欄位的結構資訊。可以用來查詢元資料。


3. 常用的 information_schema 資料表
	
	1. information_schema.schemata
		列出資料庫系統中的所有資料庫名稱。
		
	2. information_schema.tables
		列出所有資料庫中的表或視圖，包含表的名稱、表類型（table/view）和它們所屬的資料庫。
	
	3. information_schema.columns
		列出資料表中的所有欄位資訊，包括欄位名稱、資料類型、是否允許NULL、預設值等。
		
	4. information_schema.table_constraints
		列出每張表的約束條件，包括主鍵、外鍵、唯一性約束等。
		
	5. information_schema.key_column_usage
		提供有關使用主鍵和外鍵的欄位資訊，包括哪些欄位被用作鍵（鍵欄位）。

*/


-- 1. information_schema.schemata
SELECT schema_name -- 取得所有資料庫名稱。
FROM information_schema.schemata;


-- 2. information_schema.tables
SELECT table_schema,  -- 資料庫的名稱（即schema）
	   table_name,  -- 資料表的名稱。
	   table_type -- 表的類型，通常是BASE TABLE（普通表）或 VIEW（視圖）
FROM information_schema.tables
WHERE table_schema IN ('imdb', 'information_schema')
LIMIT 10;


-- 3. information_schema.columns
SELECT table_schema,  -- 資料庫名
	   table_name,  -- 資料表名
	   column_name	-- 欄位名
FROM information_schema.COLUMNS
WHERE table_schema = 'imdb' AND
	  table_name = 'movies';

-- 暸解一個資料表的外型 (m, n)
SELECT count(*) AS m -- m (行row) 
FROM imdb.movies;

SELECT count(*) AS n
FROM information_schema.COLUMNS
WHERE table_schema = 'imdb' AND
	  table_name = 'movies';


-- 4.information_schema.table_constraints
SELECT table_schema,
	   table_name,
	   constraint_type -- 約束的類型，例如 PRIMARY KEY（主鍵）、FOREIGN KEY（外鍵）、UNIQUE（唯一性約束）
FROM information_schema.table_CONSTRAINTs
WHERE table_schema = 'imdb';

	  
-- 5. information_schema.key_column_usage
SELECT table_schema,
	   table_name,
	   constraint_name, -- 設定的約束的名稱(例如：fk_movies_actors_actors)
	   referenced_table_name, -- 被引用的資料表名稱（例如：actors）
	   referenced_column_name -- 被引用的欄位名稱（例如：id）。
FROM information_schema.key_column_usage
WHERE table_schema = 'imdb';
	   
	  



	 
 
 


























































