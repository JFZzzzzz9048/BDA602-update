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
        atBat as home_AB,
        hit as home_hit,
        strikeout as home_strikeout,
        field_error as home_field_error
	    FROM team_batting_counts
        where homeTeam = 1
	    ORDER BY game_id;


DROP TEMPORARY TABLE IF EXISTS away_batting;
CREATE TEMPORARY TABLE IF NOT EXISTS away_batting AS
    SELECT Game_ID,
        atBat as away_AB,
        hit as away_hit,
        strikeout as away_strikeout,
        field_error as away_field_error
	    FROM team_batting_counts
        where homeTeam = 0
	    ORDER BY game_id;



DROP TEMPORARY TABLE IF EXISTS pitcher_info;
CREATE TEMPORARY TABLE IF NOT EXISTS pitcher_info AS
    SELECT A.Game_ID,
        A.home_team_id,
        A.away_team_id,
        SUM(A.home_hit)/SUM(A.home_AB) AS home_BA,
        SUM(B.away_hit)/SUM(B.away_AB) AS away_BA,
        A.home_strikeout AS home_strikeout,
        B.away_strikeout AS away_strikeout,
        A.home_field_error as home_field_error,
        B.away_field_error as away_field_error
	    FROM home_batting A
	    JOIN away_batting B
	    ON A.game_id = B.game_id
        GROUP BY home_team_id
	    ORDER BY A.game_id;



DROP TEMPORARY TABLE IF EXISTS starting_pitcher_info;
CREATE TEMPORARY TABLE IF NOT EXISTS starting_pitcher_info AS
    select game_id,
        toBase,
        pitchesThrown
    from  pitcher_counts
    where homeTeam = 1 and startingPitcher = 1
    order by game_id;


DROP TABLE IF EXISTS baseball_ready;
CREATE TABLE IF NOT EXISTS baseball_ready AS
    SELECT A.*,
        B.home_runs,
        B.away_runs,
        C.toBase,
        C.pitchesThrown,
        B.game_date,
        B.winner_home_or_away
	    FROM pitcher_info A
        JOIN starting_pitcher_info C
	    ON C.game_id = A.game_id
        JOIN team_game_info B
        ON A.game_id = B.game_id
	    ORDER BY A.game_id;

