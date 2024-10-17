/*
交易(Transaction)：
將多個SQL敘述集中執行就稱為交易。

交易(Transaction)有以下特性：
1. 原子性(Atomicity)：交易的操作只會是【全部執行】or【皆未執行】。
2. 持續性(Durability)：交易完成後不會遺失操作結果。
3. 一致性(Consistency)：確保資料的完整性。
4. 隔離性(Isolation)：執行到一半的操作並不會對其他操作產生影響。
*/


# 建立新的資料庫/資料表以展示 Transaction 操作	
CREATE DATABASE tcl CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE TABLE tcl.accounts(
	id int UNSIGNED AUTO_INCREMENT,
	balance double UNSIGNED, # 一種 Float 類型，可以表示非常大的數值或非常小的數值(精度更高)
	PRIMARY KEY (id)
);

SELECT * FROM tcl.accounts;

INSERT INTO tcl.accounts (balance) VALUES
 (1000),
 (1000);
 
CREATE TABLE tcl.transfers (
	id int UNSIGNED AUTO_INCREMENT,
	from_account_id int UNSIGNED,
	to_account_id int UNSIGNED,
	amount double,
	PRIMARY KEY (id)
);

SELECT * FROM tcl.transfers;


/*
交易(Transaction)的開始與結束：
1. 交易的開始： START TRANSACTION 標註
2. 交易的結束： (無誤)COMMIT、(有誤)ROLLBACK
*/

# 第一筆交易，轉帳500元 (確認無誤，完成交易)
START TRANSACTION;

INSERT INTO tcl.transfers (from_account_id, to_account_id, amount) VALUES
	(1, 2, 500);

UPDATE tcl.accounts
SET balance = balance + 500
WHERE id = 2;

UPDATE tcl.accounts
SET balance = balance - 500
WHERE id = 1;

SELECT * FROM tcl.accounts;
SELECT * FROM tcl.transfers;

COMMIT; # 執行


# 第二筆交易，轉帳600元(產生錯誤，帳目無法勾稽)
START TRANSACTION;

INSERT INTO tcl.transfers (from_account_id, to_account_id, amount) VALUES
	(1, 2, 600);

UPDATE tcl.accounts
SET balance = balance + 600
WHERE id = 2;

UPDATE tcl.accounts    # 產生錯誤(account不能為負數(unsigned))
SET balance = balance - 600
WHERE id = 1;     

SELECT * FROM tcl.accounts;  # 帳目無法勾稽
SELECT * FROM tcl.transfers;

ROLLBACK; # 復原

SELECT * FROM tcl.accounts;
SELECT * FROM tcl.transfers;


# assignment
CREATE DATABASE transaction_control CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE transaction_control.accounts(
	id int UNSIGNED,
	balance double UNSIGNED # 一種 Float 類型，可以表示非常大的數值或非常小的數值(精度更高)
);


CREATE TABLE transaction_control.transfers (
	id int UNSIGNED,
	from_account_id int UNSIGNED,
	to_account_id int UNSIGNED,
	amount double
);

INSERT INTO transaction_control.accounts (id, balance) VALUES
 (1, 1000),
 (2, 1000);

START TRANSACTION;
INSERT INTO transaction_control.transfers (id, from_account_id, to_account_id, amount) VALUES
 (1, 1, 2, 450);

UPDATE transaction_control.accounts
SET balance = balance + 450
WHERE id = 2;

UPDATE transaction_control.accounts    # 產生錯誤(account不能為負數(unsigned))
SET balance = balance - 450
WHERE id = 1;     
COMMIT;



