DROP TEMPORARY TABLE IF EXISTS team_game_info;
CREATE TEMPORARY TABLE IF NOT EXISTS team_game_info AS
	SELECT G.Game_ID AS Game_ID, G.home_team_id AS home_team_id, G.away_team_id AS away_team_id, date(G.local_date) AS game_date, B.winner_home_or_away AS winner_home_or_away,
    B.home_runs AS home_runs, B.away_runs AS away_runs
	FROM game G
	JOIN boxscore B
	ON G.game_id = B.game_id
	ORDER BY G.game_id;



DROP TEMPORARY TABLE IF EXISTS home_batting;
CREATE TEMPORARY TABLE IF NOT EXISTS home_batting AS
    SELECT Game_ID,
        team_id AS home_team_id,
        opponent_team_id AS away_team_id,
        inning AS home_inning,
        plateApperance as home_PA,
        atBat as home_AB,
        hit as home_hit,
        walk as home_walk,
        strikeout as home_strikeout
	    FROM team_batting_counts
        where homeTeam = 1
	    ORDER BY game_id;


DROP TEMPORARY TABLE IF EXISTS away_batting;
CREATE TEMPORARY TABLE IF NOT EXISTS away_batting AS
    SELECT Game_ID,
        inning AS away_inning,
        plateApperance as away_PA,
        atBat as away_AB,
        hit as away_hit,
        walk as away_walk,
        strikeout as away_strikeout
	    FROM team_batting_counts
        where homeTeam = 0
	    ORDER BY game_id;



DROP TEMPORARY TABLE IF EXISTS pitcher_info;
CREATE TEMPORARY TABLE IF NOT EXISTS pitcher_info AS
    SELECT A.Game_ID,
        A.home_team_id,
        A.away_team_id,
        AVG(A.home_inning) AS home_inning,
        AVG(A.home_PA) as home_PA,
        AVG(A.home_AB) as home_AB,
        AVG(A.home_hit) as home_hit,
        SUM(A.home_hit)/SUM(A.home_AB) AS home_BA,
        AVG(A.home_walk) as home_walk,
        AVG(A.home_strikeout) AS home_strikeout,
        AVG(B.away_inning) as away_inning,
        AVG(B.away_PA) AS away_PA,
        AVG(B.away_AB) AS away_AB,
        AVG(B.away_hit) AS away_hit,
        SUM(B.away_hit)/SUM(B.away_AB) AS away_BA,
        AVG(B.away_walk) AS away_walk,
        AVG(B.away_strikeout) AS away_strikeout
	    FROM home_batting A
	    JOIN away_batting B
	    ON A.game_id = B.game_id
        GROUP BY home_team_id
	    ORDER BY A.game_id;


DROP TABLE IF EXISTS baseball_ready;
CREATE TABLE IF NOT EXISTS baseball_ready AS
    SELECT A.*,
        B.home_runs,
        B.away_runs,
        B.game_date,
        B.winner_home_or_away
	    FROM pitcher_info A
	    JOIN team_game_info B
	    ON A.game_id = B.game_id
	    ORDER BY A.game_id;

SELECT * FROM baseball_ready
LIMIT 20;



---------------------------
Home Starting Pitcher stats:
DROP TEMPORARY TABLE IF EXISTS home_starting_pitcher_info;
CREATE TEMPORARY TABLE IF NOT EXISTS home_starting_pitcher_info AS
    select
        game_id,
        team_id as home_team_id,
        AVG(endingInning - startingInning) AS home_sp_AVG_inning,
        AVG(strikeout) as home_sp_AVG_strikeout,
        SUM(Hit)/SUM(atBat) AS home_sp_AVG_ba
    from  pitcher_counts
    where homeTeam = 1 and startingPitcher = 1
    group by team_id
    order by game_id;

SELECT * FROM home_starting_pitcher_info
LIMIT 20;

------------------------------
Away Starting Pitcher stats:
DROP TEMPORARY TABLE IF EXISTS away_starting_pitcher_info;
CREATE TEMPORARY TABLE IF NOT EXISTS away_starting_pitcher_info AS
    select
        game_id,
        team_id as away_team_id,
        AVG(endingInning - startingInning) AS away_sp_AVG_inning,
        AVG(strikeout) as away_sp_AVG_strikeout,
        SUM(Hit)/SUM(atBat) AS away_sp_AVG_ba
    from  pitcher_counts
    where homeTeam = 0 and startingPitcher = 1
    group by team_id
    order by game_id;

SELECT * FROM away_starting_pitcher_info
LIMIT 20;



------------------------------
Starting Pitcher stats:
DROP TEMPORARY TABLE IF EXISTS starting_pitcher_info;
CREATE TEMPORARY TABLE IF NOT EXISTS starting_pitcher_info AS
    SELECT
        A.Game_ID,
        A.home_sp_AVG_inning,
        B.away_sp_AVG_inning,
        A.home_sp_AVG_strikeout,
        B.away_sp_AVG_strikeout,
        A.home_sp_AVG_ba,
        B.away_sp_AVG_ba
	    FROM home_starting_pitcher_info A
	    JOIN away_starting_pitcher_info B
	    ON A.home_team_id = B.away_team_id
	    ORDER BY A.game_id;

SELECT * FROM starting_pitcher_info
LIMIT 20;
