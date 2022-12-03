-- create a temporary table batter_info only keep useful fields: batter_id, atbat, hit, game_id, date
CREATE TEMPORARY TABLE IF NOT EXISTS BATTER_INFO AS
	SELECT BC.batter AS Batter_ID, BC.atBat AS atBat, BC.Hit AS Hit, G.game_ID AS game_ID, date(G.local_date) AS Game_Date
	FROM batter_counts BC
	JOIN game G
	ON G.game_id = BC.game_id
	ORDER BY BC.batter;


-- batting AVG = Hits/ atBat (atBat should greater than 0)
-- create a historic avg table for each player
DROP TABLE IF EXISTS HISTROIC_BATTING_AVG;

CREATE TABLE IF NOT EXISTS HISTROIC_BATTING_AVG AS
	SELECT Batter_ID, SUM(Hit)/SUM(atBat) AS HISTROIC_AVG
	FROM BATTER_INFO
	WHERE atBat > 0
	GROUP BY Batter_ID;


-- create an Annual avg table for each player by using my temporary table BATTER_INFO
DROP TABLE IF EXISTS ANNUAL_BATTING_AVG;

CREATE TABLE IF NOT EXISTS ANNUAL_BATTING_AVG AS
	SELECT Batter_ID, YEAR(Game_Date) as Year, SUM(Hit)/SUM(atBat) AS ANNUAL_AVG
	FROM BATTER_INFO
	WHERE atBat > 0
	GROUP BY Batter_ID, YEAR(Game_Date);


-- create a TEMPORARY table with all info(batter_id, game_date, atBat, Hit, 100_days_prior) needed in calculating ROLLING batting average
CREATE TEMPORARY TABLE IF NOT EXISTS ROLLING_BATTING_AVG_INFO AS
SELECT Batter_ID, Game_ID, DATE(Game_Date) as Game_Date, SUM(atBat) AS atBat, SUM(Hit) AS Hit, DATE_SUB(DATE(Game_Date), INTERVAL 100 DAY) AS 100_days_prior
FROM BATTER_INFO
WHERE atBat > 0
GROUP BY Batter_ID, DATE(Game_Date);


-- USE SQL WINDOW
-- Create a temporary rolling_batting table
-- USE SQL WINDOW to find SUM of 100 days atBat and 100 days Hit
-- unxi_timestamp (100 days = 8640000 seconds), so time between 8640000 and 1 is fine
DROP TEMPORARY TABLE IF EXISTS ROLLING_BATTING_AVG;
CREATE TEMPORARY TABLE IF NOT EXISTS ROLLING_BATTING_AVG AS
SELECT Batter_ID, Game_ID, Game_Date,
SUM(atBat) OVER (
    PARTITION BY Batter_ID
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_atBat,
SUM(Hit) OVER (
    PARTITION BY Batter_ID
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_Hit
FROM ROLLING_BATTING_AVG_INFO;


-- Create Final 100 days Rolling table
DROP TABLE IF EXISTS 100_ROLLING_BATTING_AVG;
CREATE TABLE IF NOT EXISTS 100_ROLLING_BATTING_AVG AS
SELECT Batter_ID, Game_ID, Game_Date, 100_days_Hit/100_days_atBat AS 100_Rolling_AVG
FROM ROLLING_BATTING_AVG
WHERE 100_days_atBat > 0 and game_id = 12560;

select * from 100_ROLLING_BATTING_AVG;