# 創建 Role(權限的集合) 並列出所有已存在的 MySQL User名稱
CREATE ROLE 'administrator'@'localhost',  # 定義一個Role，名為 'administrator'，並指定在 'localhost' 上有效
			'normaluser'@'localhost',  # 表示這個角色或者使用者只能從運行 MySQL 的同一台機器上進行連接
			'poweruser'@'localhost';
			
SELECT USER       # mysql.user 是 MySQL 系統管理的一部分，
FROM mysql.USER;  # 用來管理所有 user 的基本信息，這些信息是全局性的，適用於 MySQL 伺服器上的所有資料庫。



# 給予 Role 不同的操作權限。
GRANT ALL ON imdb.* TO 'administrator'@'localhost';
GRANT CREATE VIEW ON imdb.* TO 'poweruser'@'localhost';
GRANT SELECT ON imdb.* TO 'poweruser'@'localhost', 'normaluser'@'localhost';


# 查看 Role 擁有的權限
SHOW GRANTS FOR 'administrator'@'localhost';  
SHOW GRANTS FOR 'poweruser'@'localhost';
SHOW GRANTS FOR 'normaluser'@'localhost';


# 撤銷 Role 擁有的權限
REVOKE CREATE VIEW ON imdb.* FROM 'poweruser'@'localhost';


# 刪除名為 'poweruser' 的 Role
DROP ROLE 'poweruser'@'localhost';


# 在資料庫中建立一個名為 Benny 且密碼為 password_1 的 user，預設 Role 為 administrator。
CREATE USER 'Benny'@'localhost' IDENTIFIED BY 'password_1' DEFAULT ROLE 'administrator'@'localhost';


# 顯示 user Benny 所擁有的權限
SHOW GRANTS FOR 'Benny'@'localhost';


# 修改 user 'Benny' 的密碼，將其更改為 postward_new。
ALTER USER 'Benny'@'localhost' IDENTIFIED BY 'postward_new';


# 將名為 'Benny' 的 user 的 'administrator' 權限撤銷
REVOKE 'administrator'@'localhost' FROM 'Benny'@'localhost';
SHOW GRANTS FOR 'Benny'@'localhost';


# 將 Role 'poweruser'的權限給 user 'Benny'，讓 user 'Benny'擁有 poweruser 的權限
GRANT 'poweruser'@'localhost' TO 'Benny'@'localhost';


# 刪除 user 'Benny' 
DROP USER 'Benny'@'localhost';
SELECT USER FROM mysql.USER;     # 查看所有 user 訊息


# 首先建立了三個角色，每個角色被分別賦予不同的資料庫權限。接著，程式碼查詢並顯示當前所有使用者。
# 創建了三個新使用者，每個使用者設置了特定的預設角色，並顯示了這些使用者的權限。
CREATE ROLE 'administrator'@'localhost',
			'normaluser'@'localhost',
			'poweruser'@'localhost';
		
SELECT USER
FROM mysql.USER;

GRANT ALL ON covid19.* TO 'administrator'@'localhost';
GRANT SELECT ON covid19.* TO 'normaluser'@'localhost';
GRANT SELECT, CREATE VIEW ON covid19.* TO 'poweruser'@'localhost';

SHOW grants FOR 'administrator'@'localhost';
SHOW grants FOR 'normaluser'@'localhost';
SHOW grants FOR 'poweruser'@'localhost';
		
CREATE USER 'ross'@'localhost' IDENTIFIED BY 'geller' DEFAULT ROLE 'administrator'@'localhost';
CREATE USER 'joey'@'localhost' IDENTIFIED BY 'tribbiani' DEFAULT ROLE 'normaluser'@'localhost';
CREATE USER 'chandler'@'localhost' IDENTIFIED BY 'bing' DEFAULT ROLE 'poweruser'@'localhost';

SHOW GRANTS FOR 'ross'@'localhost';
SHOW GRANTS FOR 'joey'@'localhost';
SHOW GRANTS FOR 'chandler'@'localhost';






























