CREATE DATABASE imdb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

ALTER TABLE imdb.actors ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE imdb.directors ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE imdb.movies ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE imdb.writers ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE imdb.movies_actors ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE imdb.movies_directors ADD CONSTRAINT PRIMARY KEY (id);
ALTER TABLE imdb.movies_writers ADD CONSTRAINT PRIMARY KEY (id);


ALTER TABLE imdb.movies_actors
ADD CONSTRAINT fk_movies_actors_actors FOREIGN KEY (actor_id) REFERENCES actors (id),
ADD CONSTRAINT fk_movies_actors_movies FOREIGN KEY (movie_id) REFERENCES movies (id);
    
ALTER TABLE imdb.movies_directors
ADD CONSTRAINT fk_movies_directors_directors FOREIGN KEY (director_id) REFERENCES directors (id),
ADD CONSTRAINT fk_movies_directors_movies FOREIGN KEY (movie_id) REFERENCES movies (id);
    
ALTER TABLE imdb.movies_writers
ADD CONSTRAINT fk_movies_writers_writers FOREIGN KEY (writer_id) REFERENCES writers (id),
ADD CONSTRAINT fk_movies_writers_movies FOREIGN KEY (movie_id) REFERENCES movies (id);
    
   
 /*
接著使用 MySQL Workbench
1. 繪製實體關係圖。
2. 匯出資料庫。 

 */
   





