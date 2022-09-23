-- create a temporary table batter_info only keep useful fields: batter_id, atbat, hit, game_id, date
CREATE TEMPORARY TABLE IF NOT EXISTS BATTER_INFO AS
	SELECT BC.batter AS Batter_ID, BC.atBat AS atBat, BC.Hit AS Hit, G.game_ID AS game_ID, date(G.local_date) AS Game_Date
	FROM batter_counts BC
	JOIN game G
	ON G.game_id = BC.game_id
	ORDER BY BC.batter;

SELECT * FROM Batter_info
LIMIT 5;


-- batting AVG = Hits/ atBat (atBat should greater than 0)
-- create a historic avg table for each player
DROP TABLE IF EXISTS HISTROIC_BATTING_AVG;

CREATE TABLE IF NOT EXISTS HISTROIC_BATTING_AVG AS
	SELECT Batter_ID, SUM(Hit)/SUM(atBat) AS HISTROIC_AVG
	FROM BATTER_INFO
	WHERE atBat > 0
	GROUP BY Batter_ID;

SELECT * FROM HISTROIC_BATTING_AVG
LIMIT 5;

"""
CREATE TABLE IF NOT EXISTS HISTROIC_BATTING_AVG_1 AS
	SELECT batter, SUM(Hit)/SUM(atBat) AS HISTROIC_AVG
	FROM batter_counts
	WHERE atBat > 0
	GROUP BY batter;
"""

-- create an Annual avg table for each player by using my temporary table
DROP TABLE IF EXISTS ANNUAL_BATTING_AVG;

CREATE TABLE IF NOT EXISTS ANNUAL_BATTING_AVG AS
	SELECT Batter_ID, YEAR(Game_Date) as Year, SUM(Hit)/SUM(atBat) AS ANNUAL_AVG
	FROM BATTER_INFO
	WHERE atBat > 0
	GROUP BY Batter_ID, YEAR(Game_Date);

SELECT * FROM ANNUAL_BATTING_AVG
LIMIT 5;

"""
# create a table with all info needed in calculating Annual batting average
DROP TABLE IF EXISTS ANNUAL_BATTING_AVG_INFO;

CREATE TABLE IF NOT EXISTS ANNUAL_BATTING_AVG_INFO AS
SELECT Batter_ID, YEAR(Game_Date) as Year, SUM(atBat) AS atBat, SUM(Hit) AS Hit
FROM BATTER_INFO
GROUP BY Batter_ID, YEAR(Game_Date);


SELECT Batter_ID, Year, Hit/atBat AS ANNUAL_AVG
FROM ANNUAL_BATTING_AVG_INFO
WHERE atBat > 0
GROUP BY Batter_ID, Year;
"""

-- create a TEMPORARY table with all info(batter_id, game_date, atBat, Hit, 100_days_prior) needed in calculating ROLLING batting average
CREATE TEMPORARY TABLE IF NOT EXISTS ROLLING_BATTING_AVG_INFO AS
SELECT Batter_ID, Game_ID, DATE(Game_Date) as Game_Date, SUM(atBat) AS atBat, SUM(Hit) AS Hit, DATE_SUB(DATE(Game_Date), INTERVAL 100 DAY) AS 100_days_prior
FROM BATTER_INFO
WHERE atBat > 0
GROUP BY Batter_ID, DATE(Game_Date);


SELECT * FROM ROLLING_BATTING_AVG_INFO
LIMIT 5;


-- create 100 days rolling table (First Edition slow...)
-- Subquery(Slow)
DROP TABLE IF EXISTS ROLLING_BATTING_AVG;

CREATE TABLE IF NOT EXISTS ROLLING_BATTING_AVG AS
SELECT
    R1.Batter_ID,
    R1.Game_Date,
    R1.Game_ID,
	R1.100_days_prior,
    (
        SELECT SUM(R2.Hit)/SUM(R2.atBat)
        FROM ROLLING_BATTING_AVG_INFO R2
        WHERE
			R2.atBat > 0
            AND R2.Batter_ID = R1.Batter_ID
            AND R2.Game_Date BETWEEN R1.100_days_prior and DATE_SUB(DATE(R1.Game_Date), INTERVAL 1 DAY)
    ) AS 100_Rolling_AVG
	FROM ROLLING_BATTING_AVG_INFO AS R1
	GROUP BY Batter_ID, Game_ID
	ORDER BY Batter_ID, Game_ID;

"""
-- USE SQL WINDOW (Still in Progress)
DROP TABLE IF EXISTS ROLLING_BATTING_AVG_1;

CREATE TABLE IF NOT EXISTS ROLLING_BATTING_AVG_1 AS
SELECT Batter_ID, Game_ID, Game_Date,
SUM(atBat) OVER (
    PARTITION BY Batter_ID
    ORDER BY Game_Date ASC
    RANGE BETWEEN INTERVAL 100 DAY PRECEDING AND CURRENT ROW
  ) AS 100_days_atBat,
SUM(Hit) OVER (
    PARTITION BY Batter_ID
    ORDER BY Game_Date ASC
    RANGE BETWEEN INTERVAL 100 DAY PRECEDING AND CURRENT ROW
  ) AS 100_days_Hit
FROM ROLLING_BATTING_AVG_INFO limit 10;
GROUP BY Batter_ID, Game_Date;
"""

SELECT
    R1.Batter_ID,
    R1.Game_Date,
    R1.Game_ID,
	R1.100_days_prior,
    (SELECT SUM(R2.Hit)/SUM(R2.atBat)
        FROM ROLLING_BATTING_AVG_INFO R2
        WHERE
			R2.atBat > 0
            AND R2.Batter_ID = R1.Batter_ID
            AND R2.Game_Date BETWEEN R1.100_days_prior and DATE_SUB(DATE(R1.Game_Date), INTERVAL 1 DAY)
    ) AS 100_Rolling_AVG
	FROM ROLLING_BATTING_AVG_INFO AS R1
	limit 100;

