DROP TEMPORARY TABLE IF EXISTS team_game_info;
CREATE TEMPORARY TABLE IF NOT EXISTS team_game_info AS
	SELECT G.Game_ID AS Game_ID, G.home_team_id AS home_team_id, G.away_team_id AS away_team_id, date(G.local_date) AS game_date, B.winner_home_or_away AS winner_home_or_away,
    B.home_runs AS home_runs, B.away_runs AS away_runs
	FROM game G
	JOIN boxscore B
	ON G.game_id = B.game_id
	ORDER BY G.game_id;


DROP TEMPORARY TABLE IF EXISTS away_batting;
CREATE TEMPORARY TABLE IF NOT EXISTS away_batting AS
    SELECT Game_ID,
        team_id AS home_team_id,
        opponent_team_id AS away_team_id,
        atBat as away_AB,
        hit as away_hit,
        hit/atBat as away_BA,
        walk as away_walk,
        toBase as away_toBase,
        plateApperance as away_PA,
        strikeout as away_strikeout,
        field_error as away_field_error,
        triple as away_triple,
        team_batting_counts.double as away_double,
        Hit_By_Pitch as away_hit_by_pitch,
        sac_fly as away_SF,
        home_run as away_home_run,
        (hit+walk+hit_by_pitch)/(atBat+walk+hit_by_pitch+sac_fly) as away_OBP,
        (hit+team_batting_counts.double+2*triple+3*home_run) as away_TB
	    FROM team_batting_counts
        where homeTeam = 1 and atBat > 0
	    ORDER BY game_id;



DROP TEMPORARY TABLE IF EXISTS home_batting;
CREATE TEMPORARY TABLE IF NOT EXISTS home_batting AS
    SELECT Game_ID,
        atBat as home_AB,
        hit as home_hit,
        hit/atBat as home_BA,
        walk as home_walk,
        toBase as home_toBase,
        plateApperance as home_PA,
        strikeout as home_strikeout,
        field_error as home_field_error,
        triple as home_triple,
        team_batting_counts.double as home_double,
        Hit_By_Pitch as home_hit_by_pitch,
        sac_fly as home_SF,
        home_run as home_home_run,
        (hit+walk+hit_by_pitch)/(atBat+walk+hit_by_pitch+sac_fly) as home_OBP,
        (hit+team_batting_counts.double+2*triple+3*home_run) as home_TB
	    FROM team_batting_counts
        where homeTeam = 0 and atBat > 0
	    ORDER BY game_id;



DROP TEMPORARY TABLE IF EXISTS pitcher_info;
CREATE TEMPORARY TABLE IF NOT EXISTS pitcher_info AS
    SELECT
        A.*,
        B.home_AB,
        B.home_hit,
        B.home_BA,
        B.home_walk,
        B.home_toBase,
        B.home_PA,
        B.home_strikeout,
        B.home_field_error,
        B.home_triple,
        B.home_double,
        B.home_hit_by_pitch,
        B.home_SF,
        B.home_home_run,
        B.home_OBP,
        B.home_TB
	    FROM away_batting A
	    JOIN home_batting B
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


DROP TABLE IF EXISTS features;
CREATE TABLE IF NOT EXISTS features AS
SELECT
home_team_id,
away_team_id,
Game_ID,
Game_Date,
AVG(home_ab) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_atBat,
AVG(home_hit) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_hit,
AVG(away_ab) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_atBat,
AVG(away_hit) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_hit,
AVG(home_BA) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_BA,
AVG(away_BA) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_BA,
AVG(home_strikeout) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_strikeout,
AVG(away_strikeout) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_strikeout,
AVG(home_field_error) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_field_error,
AVG(away_field_error) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_field_error,
AVG(home_runs) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_runs,
AVG(away_runs) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_runs,
AVG(home_walk) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_walk,
AVG(away_walk) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_walk,
AVG(home_toBase) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_toBase,
AVG(away_toBase) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_toBase,
AVG(home_PA) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_PA,
AVG(away_PA) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_PA,
AVG(home_triple) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_triple,
AVG(away_triple) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_triple,
AVG(home_hit_by_pitch) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_hit_by_pitch,
AVG(away_hit_by_pitch) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_hit_by_pitch,
AVG(home_SF) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_SF,
AVG(away_SF) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_SF,
AVG(home_OBP) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_OBP,
AVG(away_OBP) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_OBP,
AVG(home_double) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_double,
AVG(away_double) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_double,
AVG(home_home_run) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_home_run,
AVG(away_home_run) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_home_run,
AVG(home_TB) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_home_TB,
AVG(away_TB) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 8640000 PRECEDING AND 1 PRECEDING
  ) AS 100_days_away_TB,
AVG(home_ab) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_atBat,
AVG(home_hit) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_hit,
AVG(away_ab) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_atBat,
AVG(away_hit) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_hit,
AVG(home_BA) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_BA,
AVG(away_BA) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_BA,
AVG(home_strikeout) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_strikeout,
AVG(away_strikeout) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_strikeout,
AVG(home_field_error) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_field_error,
AVG(away_field_error) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_field_error,
AVG(home_runs) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_runs,
AVG(away_runs) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_runs,
AVG(home_walk) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_walk,
AVG(away_walk) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_walk,
AVG(home_toBase) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_toBase,
AVG(away_toBase) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_toBase,
AVG(home_PA) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_PA,
AVG(away_PA) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_PA,
AVG(home_triple) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_triple,
AVG(away_triple) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_triple,
AVG(home_hit_by_pitch) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_hit_by_pitch,
AVG(away_hit_by_pitch) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_hit_by_pitch,
AVG(home_SF) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_SF,
AVG(away_SF) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_SF,
AVG(home_OBP) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_OBP,
AVG(away_OBP) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_OBP,
AVG(home_double) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_double,
AVG(away_double) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_double,
AVG(home_home_run) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_home_run,
AVG(away_home_run) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_home_run,
AVG(home_TB) OVER (
    PARTITION BY home_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_home_TB,
AVG(away_TB) OVER (
    PARTITION BY away_team_id
    ORDER BY unix_timestamp(Game_Date)
    RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING
  ) AS 60_days_away_TB


FROM baseball_ready
order by game_id;


DROP TABLE IF EXISTS final_features;
CREATE TABLE IF NOT EXISTS final_features AS
select
    A.*,
    B.winner_home_or_away
    from features A
    JOIN team_game_info B
    ON A.game_id = B.game_id
    order by game_id;


DROP TABLE IF EXISTS 100_final_features;
CREATE TABLE IF NOT EXISTS 100_final_features AS
select
    Game_Date, home_team_id, away_team_id, Game_ID,
    100_days_home_hit, 100_days_away_atBat, 100_days_away_hit, 100_days_home_BA, 100_days_away_BA, 100_days_home_strikeout,
    100_days_away_strikeout, 100_days_home_field_error, 100_days_away_field_error, 100_days_home_runs, 100_days_away_runs,
    100_days_home_walk, 100_days_away_walk, 100_days_home_toBase, 100_days_away_toBase, 100_days_home_PA, 100_days_away_PA,
    100_days_home_triple, 100_days_away_triple, 100_days_home_hit_by_pitch, 100_days_away_hit_by_pitch, 100_days_home_SF,
    100_days_away_SF, 100_days_home_OBP, 100_days_away_OBP, 100_days_home_double, 100_days_away_double, 100_days_home_home_run,
    100_days_away_home_run, 100_days_home_TB, 100_days_away_TB, winner_home_or_away
    from final_features
    order by Game_ID;


DROP TABLE IF EXISTS 60_final_features;
CREATE TABLE IF NOT EXISTS 60_final_features AS
select
    Game_Date, home_team_id, away_team_id, Game_ID,
    60_days_home_hit, 60_days_away_atBat, 60_days_away_hit, 60_days_home_BA, 60_days_away_BA, 60_days_home_strikeout,
    60_days_away_strikeout, 60_days_home_field_error, 60_days_away_field_error, 60_days_home_runs, 60_days_away_runs,
    60_days_home_walk, 60_days_away_walk, 60_days_home_toBase, 60_days_away_toBase, 60_days_home_PA, 60_days_away_PA,
    60_days_home_triple, 60_days_away_triple, 60_days_home_hit_by_pitch, 60_days_away_hit_by_pitch, 60_days_home_SF,
    60_days_away_SF, 60_days_home_OBP, 60_days_away_OBP, 60_days_home_double, 60_days_away_double, 60_days_home_home_run,
    60_days_away_home_run, 60_days_home_TB, 60_days_away_TB, winner_home_or_away
    from final_features
    order by Game_ID;