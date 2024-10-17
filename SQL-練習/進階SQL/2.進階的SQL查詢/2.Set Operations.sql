/*
集合運算(Set Operation)
用來將兩個或多個查詢結果集進行合併或比較。常見的集合運算包括：
	
	1. UNION：合併兩個查詢的結果集，移除重複的資料，只保留唯一的紀錄。
	2. UNION ALL：與UNION類似，但不會移除重複的資料，會保留所有紀錄。
	3. INTERSECT：返回兩個查詢結果集中共同擁有的紀錄（交集部分）。
	4. EXCEPT：返回第一個查詢結果集中有但第二個查詢結果集中沒有的紀錄（差集）。
*/


# UNION：保留重複唯一值 ('Steven Spielberg', 'Christopher Nolan', 'Aamir Khan', 'Tom Hanks', 'Tom Cruise')
SELECT name
FROM imdb.directors
WHERE name IN ('Steven Spielberg', 'Christopher Nolan', 'Aamir Khan')
UNION 
SELECT name
FROM imdb.actors
WHERE name IN ('Tom Hanks', 'Tom Cruise', 'Aamir Khan');


# UNION ALL：保留所有值，可能會有重複值 ('Steven Spielberg', 'Christopher Nolan', 'Aamir Khan', 'Tom Hanks', 'Tom Cruise', 'Aamir Khan') 
SELECT name
FROM imdb.directors
WHERE name IN ('Steven Spielberg', 'Christopher Nolan', 'Aamir Khan')
UNION ALL 
SELECT name
FROM imdb.actors
WHERE name IN ('Tom Hanks', 'Tom Cruise', 'Aamir Khan');


# INTERSECT：保留交集值 ('Aamir Khan')
SELECT name
FROM imdb.directors
WHERE name IN ('Steven Spielberg', 'Christopher Nolan', 'Aamir Khan')
INTERSECT 
SELECT name
FROM imdb.actors
WHERE name IN ('Tom Hanks', 'Tom Cruise', 'Aamir Khan');


# EXCEPT：差集，第一個結果有，但第二個沒有的值 ('Steven Spielberg', 'Christopher Nolan')
SELECT name
FROM imdb.directors
WHERE name IN ('Steven Spielberg', 'Christopher Nolan', 'Aamir Khan')
EXCEPT  
SELECT name
FROM imdb.actors
WHERE name IN ('Tom Hanks', 'Tom Cruise', 'Aamir Khan');

