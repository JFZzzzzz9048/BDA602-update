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
        A.home_inning,
        A.home_PA,
        A.home_AB,
        A.home_hit,
        A.home_walk,
        A.home_strikeout,
        B.away_inning,
        B.away_PA,
        B.away_AB,
        B.away_hit,
        B.away_walk,
        B.away_strikeout
	    FROM home_batting A
	    JOIN away_batting B
	    ON A.game_id = B.game_id
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
